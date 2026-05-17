using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmAsrSmokeTestRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM ASRSmoke]";
        private const string StatusFileName = "LiteRtLmAsrSmokeTest.status.txt";
        private const string ConfigFileName = "LiteRtLmAsrSmokeTest.config.json";

        [SerializeField] private string modelPath = "parakeet_tdt_0.6b_v3_5s_i8.tflite";
        [SerializeField] private string audioPath = "Tactical Evaluation Results Report - March 5, 2025.mp3";
        [SerializeField] private string tokenizerJsonPath = "parakeet-tdt-0.6b-v3/tokenizer.json";
        [SerializeField] private string backend = "GPU_FP16";
        [SerializeField] private string asrMode = "parakeet";
        [SerializeField] private string asrLanguage = "auto";
        [SerializeField] private int benchmarkRuns = 1;

        private LiteRtLmUnityClient client;

        private void Start()
        {
            Application.runInBackground = true;
            client = new LiteRtLmUnityClient();
            StartCoroutine(RunSmokeTest());
        }

        private void OnDestroy()
        {
            client?.Dispose();
            client = null;
        }

        private IEnumerator RunSmokeTest()
        {
            ApplyRuntimeConfigOverrides();
            WriteStatus("START", $"mode={asrMode}, backend={backend}, language={asrLanguage}, model={modelPath}, audio={audioPath}, benchmarkRuns={benchmarkRuns}, platform={Application.platform}");

#if UNITY_ANDROID && !UNITY_EDITOR
            string resolvedModelPath = null;
            Exception resolveModelError = null;
            yield return ResolveStreamingAssetPath(
                modelPath,
                "model",
                path => resolvedModelPath = path,
                ex => resolveModelError = ex);

            if (resolveModelError != null)
            {
                WriteFailure(resolveModelError);
                yield break;
            }

            if (IsWhisperGpuRequested())
            {
                var encoderCompanionPath = GetWhisperEncoderCompanionPath(modelPath);
                if (string.IsNullOrWhiteSpace(encoderCompanionPath))
                {
                    WriteFailure(new InvalidOperationException($"Whisper GPU backend requires an encoder companion model next to {modelPath}."));
                    yield break;
                }

                string resolvedEncoderCompanionPath = null;
                Exception resolveEncoderCompanionError = null;
                yield return ResolveStreamingAssetPath(
                    encoderCompanionPath,
                    "encoder_model",
                    path => resolvedEncoderCompanionPath = path,
                    ex => resolveEncoderCompanionError = ex);

                if (resolveEncoderCompanionError != null)
                {
                    WriteFailure(resolveEncoderCompanionError);
                    yield break;
                }

                WriteStatus("ENCODER_MODEL_READY", DescribeFile(resolvedEncoderCompanionPath));

                var nativeEncoderAliasPath = GetWhisperLegacyEncoderCompanionPath(resolvedModelPath);
                if (!string.IsNullOrWhiteSpace(nativeEncoderAliasPath) &&
                    !string.Equals(nativeEncoderAliasPath, resolvedEncoderCompanionPath, StringComparison.Ordinal))
                {
                    var nativeEncoderAliasDirectory = Path.GetDirectoryName(nativeEncoderAliasPath);
                    if (!string.IsNullOrWhiteSpace(nativeEncoderAliasDirectory))
                    {
                        Directory.CreateDirectory(nativeEncoderAliasDirectory);
                    }

                    var shouldCopyAlias = !File.Exists(nativeEncoderAliasPath) ||
                        new FileInfo(nativeEncoderAliasPath).Length != new FileInfo(resolvedEncoderCompanionPath).Length;
                    if (shouldCopyAlias)
                    {
                        File.Copy(resolvedEncoderCompanionPath, nativeEncoderAliasPath, true);
                    }

                    WriteStatus("ENCODER_NATIVE_ALIAS_READY", DescribeFile(nativeEncoderAliasPath));
                }
            }

            string resolvedAudioPath = null;
            Exception resolveAudioError = null;
            yield return ResolveStreamingAssetPath(
                audioPath,
                "audio",
                path => resolvedAudioPath = path,
                ex => resolveAudioError = ex);

            if (resolveAudioError != null)
            {
                WriteFailure(resolveAudioError);
                yield break;
            }

            string resolvedTokenizerPath = null;
            Exception resolveTokenizerError = null;
            yield return ResolveStreamingAssetPath(
                tokenizerJsonPath,
                "tokenizer",
                path => resolvedTokenizerPath = path,
                ex => resolveTokenizerError = ex);

            if (resolveTokenizerError != null)
            {
                WriteFailure(resolveTokenizerError);
                yield break;
            }

            RunAsrWithStatus(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath);
#else
            WriteStatus("SKIP", "ASR smoke test runner only executes in Android player builds.");
            yield break;
#endif
        }

        private void RunAsrWithStatus(string resolvedModelPath, string resolvedAudioPath, string resolvedTokenizerPath)
        {
            try
            {
                WriteStatus("AUDIO_READY", DescribeFile(resolvedAudioPath));
                WriteStatus("MODEL_READY", DescribeFile(resolvedModelPath));
                WriteStatus("TOKENIZER_READY", DescribeFile(resolvedTokenizerPath));

                var inspectionJson = client.InspectLiteRtModel(resolvedModelPath);
                WriteStatus("INSPECT_RESULT", OneLine(Truncate(inspectionJson, 3000)));

                var normalizedMode = string.IsNullOrWhiteSpace(asrMode)
                    ? "parakeet"
                    : asrMode.Trim().ToLowerInvariant();
                var normalizedLanguage = string.IsNullOrWhiteSpace(asrLanguage)
                    ? "auto"
                    : asrLanguage.Trim().ToLowerInvariant();
                var runs = Math.Max(1, benchmarkRuns);
                var totalElapsedSeconds = 0.0;
                var totalCompileSeconds = 0.0;
                var totalEncodeSeconds = 0.0;
                var totalDecodeSeconds = 0.0;
                var compileCount = 0;
                var encodeCount = 0;
                var decodeCount = 0;
                var lastAsrJson = string.Empty;

                for (var run = 1; run <= runs; run++)
                {
                    var startedAt = Time.realtimeSinceStartup;
                    WriteStatus("ASR_INVOKE", $"run={run}/{runs}, invoking {normalizedMode} LiteRT ASR smoke path with backend={backend}, language={normalizedLanguage}.");
                    var asrJson = normalizedMode == "whisper"
                        ? client.RunWhisperAsrSmoke(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath, backend, normalizedLanguage)
                        : client.RunParakeetAsrSmoke(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath, backend);
                    var elapsedSeconds = Time.realtimeSinceStartup - startedAt;

                    if (string.IsNullOrWhiteSpace(asrJson))
                    {
                        throw new InvalidOperationException($"{normalizedMode} ASR smoke test returned an empty result on run {run}/{runs}.");
                    }
                    if (asrJson.Contains("\"success\": false", StringComparison.OrdinalIgnoreCase))
                    {
                        throw new InvalidOperationException($"{normalizedMode} ASR smoke test reported failure on run {run}/{runs}: {OneLine(Truncate(asrJson, 1000))}");
                    }

                    totalElapsedSeconds += elapsedSeconds;
                    if (TryGetJsonDouble(asrJson, "compileSeconds", out var compileSeconds))
                    {
                        totalCompileSeconds += compileSeconds;
                        compileCount++;
                    }
                    if (TryGetJsonDouble(asrJson, "encodeSeconds", out var encodeSeconds))
                    {
                        totalEncodeSeconds += encodeSeconds;
                        encodeCount++;
                    }
                    if (TryGetJsonDouble(asrJson, "decodeSeconds", out var decodeSeconds))
                    {
                        totalDecodeSeconds += decodeSeconds;
                        decodeCount++;
                    }

                    lastAsrJson = asrJson;
                    WriteStatus("ASR_RESULT", $"run={run}/{runs}, elapsedSeconds={elapsedSeconds:0.###}, raw={OneLine(Truncate(asrJson, 3000))}");
                }

                WriteStatus(
                    "ASR_BENCHMARK_SUMMARY",
                    $"runs={runs}, averageElapsedSeconds={(totalElapsedSeconds / runs):0.###}, averageCompileSeconds={FormatAverage(totalCompileSeconds, compileCount)}, averageEncodeSeconds={FormatAverage(totalEncodeSeconds, encodeCount)}, averageDecodeSeconds={FormatAverage(totalDecodeSeconds, decodeCount)}");
                WriteStatus("SUCCESS", $"runs={runs}, elapsedSeconds={totalElapsedSeconds:0.###}, averageElapsedSeconds={(totalElapsedSeconds / runs):0.###}, resultLength={lastAsrJson.Length}");
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
            }
        }

        private IEnumerator ResolveStreamingAssetPath(
            string configuredPath,
            string label,
            Action<string> onSuccess,
            Action<Exception> onError)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
            {
                onError(new ArgumentException($"{label} path is required."));
                yield break;
            }

            if (Path.IsPathRooted(configuredPath))
            {
                if (!File.Exists(configuredPath))
                {
                    onError(new FileNotFoundException($"{label} file not found: {configuredPath}", configuredPath));
                    yield break;
                }

                onSuccess(configuredPath);
                yield break;
            }

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "ASR");
            Directory.CreateDirectory(destinationDirectory);
            var destinationPath = Path.Combine(destinationDirectory, configuredPath);
            var destinationParent = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(destinationParent))
            {
                Directory.CreateDirectory(destinationParent);
            }

            if (File.Exists(destinationPath))
            {
                WriteStatus($"{label.ToUpperInvariant()}_READY", $"{DescribeFile(destinationPath)}, copied=False");
                onSuccess(destinationPath);
                yield break;
            }

            var sourcePath = Path.Combine(Application.streamingAssetsPath, configuredPath).Replace("\\", "/");
            WriteStatus($"COPY_{label.ToUpperInvariant()}", $"source={sourcePath}, destination={destinationPath}");

            using var request = UnityWebRequest.Get(sourcePath);
            request.downloadHandler = new DownloadHandlerFile(destinationPath)
            {
                removeFileOnAbort = true,
            };

            yield return request.SendWebRequest();
            if (request.result != UnityWebRequest.Result.Success)
            {
                onError(new IOException($"Failed to copy {label} from StreamingAssets: {request.error}"));
                yield break;
            }

            if (!File.Exists(destinationPath))
            {
                onError(new FileNotFoundException($"{label} copy did not create destination file: {destinationPath}", destinationPath));
                yield break;
            }

            var copiedFileInfo = new FileInfo(destinationPath);
            if (copiedFileInfo.Length <= 0)
            {
                onError(new IOException($"{label} copy produced an empty destination file: {destinationPath}"));
                yield break;
            }

            WriteStatus($"COPY_{label.ToUpperInvariant()}_DONE", $"{DescribeFile(destinationPath)}, copied=True");
            onSuccess(destinationPath);
        }

        private void WriteFailure(Exception ex)
        {
            WriteStatus("FAILURE", ex.ToString());
            Debug.LogException(ex);
        }

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
                if (TryGetJsonString(json, "modelPath", out var configuredModelPath))
                {
                    modelPath = configuredModelPath;
                }
                if (TryGetJsonString(json, "audioPath", out var configuredAudioPath))
                {
                    audioPath = configuredAudioPath;
                }
                if (TryGetJsonString(json, "tokenizerJsonPath", out var configuredTokenizerPath))
                {
                    tokenizerJsonPath = configuredTokenizerPath;
                }
                if (TryGetJsonString(json, "backend", out var configuredBackend))
                {
                    backend = configuredBackend;
                }
                if (TryGetJsonString(json, "asrMode", out var configuredAsrMode))
                {
                    asrMode = configuredAsrMode;
                }
                if (TryGetJsonString(json, "asrLanguage", out var configuredAsrLanguage))
                {
                    asrLanguage = configuredAsrLanguage;
                }
                if (TryGetJsonInt(json, "benchmarkRuns", out var configuredBenchmarkRuns) && configuredBenchmarkRuns > 0)
                {
                    benchmarkRuns = configuredBenchmarkRuns;
                }

                WriteStatus("CONFIG", $"loaded={configPath}, mode={asrMode}, model={modelPath}, backend={backend}, language={asrLanguage}, benchmarkRuns={benchmarkRuns}");
            }
            catch (Exception ex)
            {
                WriteFailure(new InvalidOperationException($"Failed to load ASR smoke runtime config: {configPath}", ex));
            }
        }

        private static void WriteStatus(string phase, string message)
        {
            var line = $"{LogPrefix} {phase}: {message}";
            Debug.Log(line);

            try
            {
                var statusPath = Path.Combine(Application.persistentDataPath, StatusFileName);
                File.AppendAllText(statusPath, $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"{LogPrefix} failed to write status file: {ex.Message}");
            }
        }

        private static string DescribeFile(string path)
        {
            var fileInfo = new FileInfo(path);
            return $"path={path}, bytes={fileInfo.Length}";
        }

        private static string Truncate(string value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
            {
                return value;
            }

            return value.Substring(0, maxLength) + "...";
        }

        private static string OneLine(string value)
        {
            return string.IsNullOrEmpty(value)
                ? string.Empty
                : value.Replace("\r", " ").Replace("\n", " ").Trim();
        }

        private static bool TryGetJsonString(string json, string propertyName, out string value)
        {
            var match = Regex.Match(json, $"\"{Regex.Escape(propertyName)}\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
            if (match.Success)
            {
                value = Regex.Unescape(match.Groups[1].Value);
                return true;
            }

            value = string.Empty;
            return false;
        }

        private static bool TryGetJsonInt(string json, string propertyName, out int value)
        {
            var match = Regex.Match(json, $"\"{Regex.Escape(propertyName)}\"\\s*:\\s*(-?[0-9]+)");
            if (match.Success && int.TryParse(match.Groups[1].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
            {
                return true;
            }

            value = 0;
            return false;
        }

        private static bool TryGetJsonDouble(string json, string propertyName, out double value)
        {
            var match = Regex.Match(json, $"\"{Regex.Escape(propertyName)}\"\\s*:\\s*(-?(?:[0-9]+\\.?[0-9]*|[0-9]*\\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
            if (match.Success && double.TryParse(match.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            {
                return true;
            }

            value = 0.0;
            return false;
        }

        private static string FormatAverage(double total, int count)
        {
            return count <= 0
                ? "N/A"
                : (total / count).ToString("0.###", CultureInfo.InvariantCulture);
        }

        private bool IsWhisperGpuRequested()
        {
            return string.Equals(asrMode, "whisper", StringComparison.OrdinalIgnoreCase)
                && !string.IsNullOrWhiteSpace(backend)
                && backend.Trim().StartsWith("GPU", StringComparison.OrdinalIgnoreCase);
        }

        private static string GetWhisperEncoderCompanionPath(string configuredModelPath)
        {
            if (string.IsNullOrWhiteSpace(configuredModelPath))
            {
                return null;
            }

            var directory = Path.GetDirectoryName(configuredModelPath);
            var fileName = Path.GetFileName(configuredModelPath);
            if (string.IsNullOrWhiteSpace(fileName))
            {
                return null;
            }

            string preferredCompanionFileName;
            string legacyCompanionFileName = null;
            if (fileName.EndsWith("_f32.tflite", StringComparison.OrdinalIgnoreCase))
            {
                preferredCompanionFileName = fileName.Substring(0, fileName.Length - ".tflite".Length) + "_encoder.tflite";
                legacyCompanionFileName = fileName.Substring(0, fileName.Length - "_f32.tflite".Length) + "_encoder_f32.tflite";
            }
            else if (fileName.EndsWith(".tflite", StringComparison.OrdinalIgnoreCase))
            {
                preferredCompanionFileName = fileName.Substring(0, fileName.Length - ".tflite".Length) + "_encoder.tflite";
            }
            else
            {
                return null;
            }

            var preferredPath = string.IsNullOrWhiteSpace(directory)
                ? preferredCompanionFileName
                : Path.Combine(directory, preferredCompanionFileName).Replace("\\", "/");
            if (File.Exists(preferredPath) || string.IsNullOrWhiteSpace(legacyCompanionFileName))
            {
                return preferredPath;
            }

            var legacyPath = string.IsNullOrWhiteSpace(directory)
                ? legacyCompanionFileName
                : Path.Combine(directory, legacyCompanionFileName).Replace("\\", "/");
            return File.Exists(legacyPath) ? legacyPath : preferredPath;
        }

        private static string GetWhisperLegacyEncoderCompanionPath(string configuredModelPath)
        {
            if (string.IsNullOrWhiteSpace(configuredModelPath))
            {
                return null;
            }

            var directory = Path.GetDirectoryName(configuredModelPath);
            var fileName = Path.GetFileName(configuredModelPath);
            if (string.IsNullOrWhiteSpace(fileName) ||
                !fileName.EndsWith("_f32.tflite", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            var legacyCompanionFileName = fileName.Substring(0, fileName.Length - "_f32.tflite".Length) + "_encoder_f32.tflite";
            return string.IsNullOrWhiteSpace(directory)
                ? legacyCompanionFileName
                : Path.Combine(directory, legacyCompanionFileName).Replace("\\", "/");
        }
    }
}
