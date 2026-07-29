using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Headless device gate for Supertonic TTS on LiteRT: speaks a list of Korean
    /// sentences, writes the per-utterance timings to a status file, and keeps the
    /// WAVs so they can be pulled back and transcribed on the desktop.
    ///
    /// It exists as an APK runner rather than an interactive scene because the
    /// question this answers — does it run on kona, and how fast — needs a
    /// repeatable log, not a screenshot. The interactive scene
    /// (LiteRtLmTtsTestScene) is for listening.
    ///
    /// Sentences and settings can be overridden at run time by pushing
    /// <c>LiteRtLmTtsSmokeTest.config.json</c> into the app's persistent data
    /// directory, so a sweep does not need a rebuild.
    ///
    /// Model layout comes from Tools/Research/Supertonic/Deploy-SupertonicLiteRt.ps1.
    /// StreamingAssets inside an APK is not a real directory, so every file the
    /// native side has to open is copied out first — the same approach the ASR
    /// smoke runner takes.
    /// </summary>
    public sealed class LiteRtLmTtsSmokeTestRunner : MonoBehaviour, ILiteRtLmModelHost
    {
        private const string LogPrefix = "[LiteRT-LM TTSSmoke]";
        private const string StatusFileName = "LiteRtLmTtsSmokeTest.status.txt";
        private const string ConfigFileName = "LiteRtLmTtsSmokeTest.config.json";

        [Header("Model set (StreamingAssets-relative)")]
        [SerializeField] private string root = "TTS/supertonic-litert";
        [SerializeField] private string voice = "F1";

        [Header("Synthesis")]
        [SerializeField] private int steps = 4;
        [SerializeField] private float speed = 1.05f;
        [SerializeField] private string backend = "CPU";
        [SerializeField] private int seed = 1234;
        [SerializeField] private string language = "ko";

        /// <summary>
        /// Deliberately spans the bucket ladder: 39, 69 and 112 ids on the desktop,
        /// which select the 64, 128 and 128 rungs. A run that only ever hits one
        /// bucket would not prove the chooser.
        /// </summary>
        [SerializeField]
        private string[] sentences =
        {
            "고도 백 미터로 상승합니다.",
            "고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트.",
            "경고. 강풍이 감지되었습니다. 고도를 낮춥니다. 귀환을 시작합니다. 예상 소요 시간 삼 분.",
        };

        [Header("Repeat")]
        [Tooltip("Runs per sentence. The first pays model compilation; later ones " +
                 "show the warm cost, which is what a real interaction sees.")]
        [SerializeField] private int runsPerSentence = 2;

        private LiteRtLmUnityClient _client;
        private LiteRtLmSupertonicText _textFrontEnd;
        private readonly Dictionary<string, string> _resolved = new Dictionary<string, string>();

        private void Start()
        {
            Application.runInBackground = true;
            _client = new LiteRtLmUnityClient();
            StartCoroutine(RunSmokeTest());
        }

        private void OnDestroy() => ReleaseModels();

        /// <inheritdoc />
        public void ReleaseModels()
        {
            _client?.Dispose();
            _client = null;
            _textFrontEnd = null;
        }

        private IEnumerator RunSmokeTest()
        {
            ApplyRuntimeConfigOverrides();
            WriteStatus(
                "START",
                $"root={root}, voice={voice}, steps={steps}, speed={speed}, backend={backend}, " +
                $"seed={seed}, sentences={sentences?.Length ?? 0}, runsPerSentence={runsPerSentence}, " +
                $"platform={Application.platform}");

#if UNITY_ANDROID && !UNITY_EDITOR
            if (sentences == null || sentences.Length == 0)
            {
                WriteStatus("FAILURE", "no sentences configured.");
                yield break;
            }

            // The front-end tables and the two dynamic graphs are shared by every
            // utterance; the bucketed pair depends on the sentence, so those are
            // resolved lazily below.
            var required = new[]
            {
                $"{root}/assets/tts.json",
                $"{root}/assets/unicode_indexer.json",
                $"{root}/assets/{voice}.json",
                $"{root}/dynamic/duration_predictor",
                $"{root}/dynamic/text_encoder",
            };
            foreach (var entry in required)
            {
                var failed = false;
                yield return Resolve(entry, error =>
                {
                    WriteFailure(error);
                    failed = true;
                });
                if (failed)
                {
                    yield break;
                }
            }

            try
            {
                _textFrontEnd = LiteRtLmSupertonicText.FromIndexerJson(
                    File.ReadAllText(_resolved[$"{root}/assets/unicode_indexer.json"]));
            }
            catch (Exception exception)
            {
                WriteFailure(exception);
                yield break;
            }

            var wavDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "TtsSmoke");
            Directory.CreateDirectory(wavDirectory);

            var failures = 0;
            var total = 0;
            for (var index = 0; index < sentences.Length; index++)
            {
                var sentence = sentences[index];
                int[] ids;
                try
                {
                    _textFrontEnd.Encode(sentence, language, out ids, out _);
                }
                catch (Exception exception)
                {
                    WriteStatus("FAIL", $"[{index + 1}] front end: {exception.Message}");
                    failures++;
                    continue;
                }

                var bucket = LiteRtLmSupertonicTts.ChooseBucket(ids.Length, ids.Length);
                if (bucket < 0)
                {
                    WriteStatus("FAIL", $"[{index + 1}] {ids.Length} ids exceed every bucket.");
                    failures++;
                    continue;
                }

                var bucketFailed = false;
                foreach (var stem in new[] { "vector_estimator", "vocoder" })
                {
                    yield return Resolve($"{root}/st-b{bucket}/{stem}", error =>
                    {
                        WriteFailure(error);
                        bucketFailed = true;
                    });
                    if (bucketFailed) { break; }
                }
                if (bucketFailed)
                {
                    failures++;
                    continue;
                }

                var padded = new int[bucket];
                Array.Copy(ids, padded, ids.Length);

                WriteStatus(
                    "SENTENCE",
                    $"[{index + 1}/{sentences.Length}] ids={ids.Length}, bucket={bucket}, text={sentence}");

                for (var run = 1; run <= Mathf.Max(1, runsPerSentence); run++)
                {
                    total++;
                    var wavPath = Path.Combine(
                        wavDirectory, $"tts-{index + 1:d2}-run{run}.wav");

                    string json = null;
                    var elapsed = Time.realtimeSinceStartup;
                    try
                    {
                        json = _client.RunSupertonicTts(
                            _resolved[$"{root}/dynamic/duration_predictor"],
                            _resolved[$"{root}/dynamic/text_encoder"],
                            _resolved[$"{root}/st-b{bucket}/vector_estimator"],
                            _resolved[$"{root}/st-b{bucket}/vocoder"],
                            _resolved[$"{root}/assets/tts.json"],
                            _resolved[$"{root}/assets/{voice}.json"],
                            padded, ids.Length, steps, speed, backend, seed, wavPath);
                    }
                    catch (Exception exception)
                    {
                        WriteStatus("FAIL", $"[{index + 1}] run {run}: {exception.Message}");
                        failures++;
                        continue;
                    }

                    elapsed = Time.realtimeSinceStartup - elapsed;
                    var reportedError = ExtractString(json, "error");
                    if (!string.IsNullOrEmpty(reportedError))
                    {
                        WriteStatus("FAIL", $"[{index + 1}] run {run}: {reportedError}");
                        failures++;
                        continue;
                    }

                    var audio = ExtractNumber(json, "audioSeconds");
                    var rtf = ExtractNumber(json, "rtf");
                    var perStep = ExtractNumber(json, "vectorEstimatorPerStepSeconds");
                    var vocoder = ExtractNumber(json, "vocoderSeconds");
                    var compile = ExtractNumber(json, "compileSeconds");
                    var cache = ExtractString(json, "compiledModelCache");
                    var bytes = File.Exists(wavPath) ? new FileInfo(wavPath).Length : 0;

                    WriteStatus(
                        "RESULT",
                        $"[{index + 1}] run {run}: rtf={rtf:0.000}, audioSeconds={audio:0.00}, " +
                        $"wallSeconds={elapsed:0.000}, vePerStep={perStep:0.000}, " +
                        $"vocoder={vocoder:0.000}, compile={compile:0.000}, cache={cache}, " +
                        $"wav={wavPath}, bytes={bytes}");

                    // Checksums of the tensors the native side actually fed, to be
                    // compared against Tools/Research/Supertonic/TtsBench/dump_supertonic_input_md5.py.
                    // A transposed tensor passes every size check, so this is the
                    // only cheap way to localize a layout mistake.
                    if (run == 1)
                    {
                        WriteStatus(
                            "CHECKSUM",
                            $"[{index + 1}] duration={ExtractNumber(json, "durationSeconds"):0.0000}, " +
                            $"ids={ExtractString(json, "idsMd5")}, " +
                            $"styleDp={ExtractString(json, "styleDpMd5")}, " +
                            $"styleTtl={ExtractString(json, "styleTtlMd5")}, " +
                            $"textMask={ExtractString(json, "textMaskMd5")}, " +
                            $"textEmb={ExtractString(json, "textEmbMd5")}, " +
                            $"embedDim={ExtractNumber(json, "embedDim"):0}, " +
                            $"latentLen={ExtractNumber(json, "latentLen"):0}, " +
                            $"idsFirst8={ExtractIntArrayText(json, "idsFirst8")}");
                    }

                    if (bytes <= 44)
                    {
                        WriteStatus("FAIL", $"[{index + 1}] run {run}: WAV is empty.");
                        failures++;
                    }

                    yield return null;
                }
            }

            WriteStatus(
                failures == 0 ? "SUCCESS" : "FAILURE",
                $"runs={total}, failures={failures}, wavDirectory={wavDirectory}");
#else
            WriteStatus("SKIP", "TTS smoke runner only executes in Android player builds.");
            yield break;
#endif
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        /// <summary>
        /// Copies one StreamingAssets entry out of the APK and records the real
        /// path. A path without an extension is treated as a graph directory and
        /// the single .tflite inside it is taken, so the precision suffix
        /// (`_float32`, `_w8`) stays a deployment choice rather than being encoded
        /// in the runner.
        /// </summary>
        private IEnumerator Resolve(string relative, Action<Exception> onError)
        {
            if (_resolved.ContainsKey(relative))
            {
                yield break;
            }

            var isGraphDirectory = string.IsNullOrEmpty(Path.GetExtension(relative));
            var candidates = new List<string>();
            if (isGraphDirectory)
            {
                // The APK has no directory listing, so the file name cannot be
                // discovered — the known variants are probed instead.
                var stem = Path.GetFileName(relative);
                foreach (var suffix in new[] { "_w8", "_float32", "_i8", "_float16" })
                {
                    candidates.Add($"{relative}/{stem}{suffix}.tflite");
                }
            }
            else
            {
                candidates.Add(relative);
            }

            Exception lastError = null;
            foreach (var candidate in candidates)
            {
                var destination = Path.Combine(
                    Application.persistentDataPath, "LiteRTLM", candidate.Replace('/', Path.DirectorySeparatorChar));
                Directory.CreateDirectory(Path.GetDirectoryName(destination));

                if (File.Exists(destination) && new FileInfo(destination).Length > 0)
                {
                    _resolved[relative] = destination;
                    WriteStatus("READY", $"{relative} -> {destination} (cached, {new FileInfo(destination).Length} bytes)");
                    yield break;
                }

                var source = Path.Combine(Application.streamingAssetsPath, candidate).Replace("\\", "/");
                using (var request = UnityWebRequest.Get(source))
                {
                    yield return request.SendWebRequest();
#if UNITY_2020_1_OR_NEWER
                    var failed = request.result != UnityWebRequest.Result.Success;
#else
                    var failed = request.isNetworkError || request.isHttpError;
#endif
                    if (failed)
                    {
                        lastError = new FileNotFoundException($"{candidate}: {request.error}", candidate);
                        continue;
                    }

                    try
                    {
                        File.WriteAllBytes(destination, request.downloadHandler.data);
                    }
                    catch (Exception exception)
                    {
                        onError(exception);
                        yield break;
                    }
                }

                _resolved[relative] = destination;
                WriteStatus("READY", $"{relative} -> {destination} ({new FileInfo(destination).Length} bytes)");
                yield break;
            }

            onError(lastError ?? new FileNotFoundException($"could not stage {relative}", relative));
        }
#endif

        private void ApplyRuntimeConfigOverrides()
        {
            var configPath = Path.Combine(Application.persistentDataPath, ConfigFileName);
            if (!File.Exists(configPath))
            {
                return;
            }

            try
            {
                var json = File.ReadAllText(configPath);
                var configuredRoot = ExtractString(json, "root");
                if (!string.IsNullOrEmpty(configuredRoot)) { root = configuredRoot; }
                var configuredVoice = ExtractString(json, "voice");
                if (!string.IsNullOrEmpty(configuredVoice)) { voice = configuredVoice; }
                var configuredBackend = ExtractString(json, "backend");
                if (!string.IsNullOrEmpty(configuredBackend)) { backend = configuredBackend; }
                var configuredLanguage = ExtractString(json, "language");
                if (!string.IsNullOrEmpty(configuredLanguage)) { language = configuredLanguage; }

                var configuredSteps = ExtractNumber(json, "steps");
                if (!double.IsNaN(configuredSteps) && configuredSteps >= 1) { steps = (int)configuredSteps; }
                var configuredSpeed = ExtractNumber(json, "speed");
                if (!double.IsNaN(configuredSpeed) && configuredSpeed > 0) { speed = (float)configuredSpeed; }
                var configuredSeed = ExtractNumber(json, "seed");
                if (!double.IsNaN(configuredSeed)) { seed = (int)configuredSeed; }
                var configuredRuns = ExtractNumber(json, "runsPerSentence");
                if (!double.IsNaN(configuredRuns) && configuredRuns >= 1) { runsPerSentence = (int)configuredRuns; }

                var configuredSentences = ExtractStringArray(json, "sentences");
                if (configuredSentences != null && configuredSentences.Length > 0)
                {
                    sentences = configuredSentences;
                }

                WriteStatus(
                    "CONFIG",
                    $"loaded={configPath}, root={root}, voice={voice}, steps={steps}, " +
                    $"backend={backend}, sentences={sentences.Length}, runsPerSentence={runsPerSentence}");
            }
            catch (Exception exception)
            {
                WriteFailure(new InvalidOperationException(
                    $"Failed to load TTS smoke runtime config: {configPath}", exception));
            }
        }

        private static string ExtractString(string json, string key) =>
            LiteRtLmSupertonicTtsJson.String(json, key);

        private static double ExtractNumber(string json, string key) =>
            LiteRtLmSupertonicTts.ExtractJsonNumber(json, key);

        /// <summary>Renders a flat numeric array as text, for the checksum line.</summary>
        private static string ExtractIntArrayText(string json, string key)
        {
            if (string.IsNullOrEmpty(json))
            {
                return "?";
            }

            var needle = "\"" + key + "\"";
            var at = json.IndexOf(needle, StringComparison.Ordinal);
            if (at < 0)
            {
                return "?";
            }

            var open = json.IndexOf('[', at);
            var close = json.IndexOf(']', open + 1);
            if (open < 0 || close < 0)
            {
                return "?";
            }

            return json.Substring(open, close - open + 1)
                .Replace("\n", string.Empty)
                .Replace("\r", string.Empty)
                .Replace(" ", string.Empty);
        }

        /// <summary>Reads a flat array of strings; null when the key is absent.</summary>
        private static string[] ExtractStringArray(string json, string key)
        {
            if (string.IsNullOrEmpty(json))
            {
                return null;
            }

            var needle = "\"" + key + "\"";
            var at = json.IndexOf(needle, StringComparison.Ordinal);
            if (at < 0)
            {
                return null;
            }

            var open = json.IndexOf('[', at);
            var close = json.IndexOf(']', open + 1);
            if (open < 0 || close < 0)
            {
                return null;
            }

            var values = new List<string>();
            var i = open + 1;
            while (i < close)
            {
                var quote = json.IndexOf('"', i);
                if (quote < 0 || quote > close)
                {
                    break;
                }

                var builder = new System.Text.StringBuilder();
                i = quote + 1;
                while (i < json.Length && json[i] != '"')
                {
                    if (json[i] == '\\' && i + 1 < json.Length)
                    {
                        builder.Append(json[i + 1] == 'n' ? '\n' : json[i + 1]);
                        i += 2;
                        continue;
                    }

                    builder.Append(json[i]);
                    i++;
                }

                values.Add(builder.ToString());
                i++;
            }

            return values.Count > 0 ? values.ToArray() : null;
        }

        private static void WriteFailure(Exception exception)
        {
            WriteStatus("FAILURE", exception.Message);
            Debug.LogException(exception);
        }

        private static void WriteStatus(string phase, string message)
        {
            Debug.Log($"{LogPrefix} {phase}: {message}");

            try
            {
                var statusPath = Path.Combine(Application.persistentDataPath, StatusFileName);
                File.AppendAllText(
                    statusPath, $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}");
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"{LogPrefix} failed to write status file: {exception.Message}");
            }
        }
    }

    /// <summary>String half of the tiny JSON reader; the numeric half lives on
    /// <see cref="LiteRtLmSupertonicTts"/> and is shared with the interactive scene.</summary>
    internal static class LiteRtLmSupertonicTtsJson
    {
        internal static string String(string json, string key)
        {
            if (string.IsNullOrEmpty(json))
            {
                return string.Empty;
            }

            var needle = "\"" + key + "\"";
            var at = json.IndexOf(needle, StringComparison.Ordinal);
            if (at < 0)
            {
                return string.Empty;
            }

            var colon = json.IndexOf(':', at + needle.Length);
            if (colon < 0)
            {
                return string.Empty;
            }

            var quote = json.IndexOf('"', colon + 1);
            if (quote < 0)
            {
                return string.Empty;
            }

            var builder = new System.Text.StringBuilder();
            for (var i = quote + 1; i < json.Length; i++)
            {
                if (json[i] == '\\' && i + 1 < json.Length)
                {
                    builder.Append(json[i + 1] == 'n' ? '\n' : json[i + 1]);
                    i++;
                    continue;
                }

                if (json[i] == '"')
                {
                    break;
                }

                builder.Append(json[i]);
            }

            return builder.ToString();
        }
    }
}
