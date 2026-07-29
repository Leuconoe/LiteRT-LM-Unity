using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Supertonic TTS running on LiteRT — the neural voice, as opposed to the
    /// platform voices in <see cref="LiteRtLmSystemTts"/>.
    ///
    /// This exists because the METALENSE2 target is an AOSP build with no TTS
    /// engine and no way to install one, so a bundled model is the only way the
    /// device speaks at all.
    ///
    /// Split of work, and why:
    ///   • Text front end in C# (<see cref="LiteRtLmSupertonicText"/>) — step one
    ///     is Unicode NFKD, and doing that natively would mean linking ICU into
    ///     the AAR.
    ///   • Bucket choice in C# — the fast graphs are converted at fixed shapes, so
    ///     the text must be padded to a converted size. Picking the smallest that
    ///     fits keeps the padding waste bounded.
    ///   • The four graphs and the whole flow-matching loop native, behind one
    ///     bridge call per utterance: one JNI crossing instead of one per step,
    ///     and the latent never leaves native memory.
    ///
    /// Layout expected under StreamingAssets (staged by
    /// Tools/Research/Supertonic/Deploy-SupertonicLiteRt.ps1):
    /// <code>
    /// TTS/supertonic-litert/
    ///   dynamic/{duration_predictor,text_encoder}/*.tflite
    ///   st-b64/{vector_estimator,vocoder}/*.tflite
    ///   st-b128/…  st-b256/…
    ///   assets/{tts.json,unicode_indexer.json,F1.json}
    /// </code>
    /// </summary>
    public sealed class LiteRtLmSupertonicTts : ILiteRtLmTts
    {
        /// <summary>Buckets the ladder is converted at, smallest first.</summary>
        public static readonly int[] Buckets = { 64, 128, 256 };

        private const string DefaultRoot = "TTS/supertonic-litert";
        private const string DefaultVoice = "F1";

        // Matches the desktop bench default: 4 steps is 1.6x faster than 8 with mel
        // correlation 0.984, and every step count transcribes correctly — so the
        // spectral measurement, not ASR, is what set this.
        private const int DefaultSteps = 4;

        private readonly string _root;
        private readonly string _voice;
        private readonly bool _ownsAudioSource;

        private AudioSource _audioSource;
        private LiteRtLmUnityClient _client;
        private LiteRtLmSupertonicText _text;
        private string _lastError = string.Empty;

        public LiteRtLmSupertonicTts(
            AudioSource audioSource = null,
            string streamingAssetsRoot = DefaultRoot,
            string voice = DefaultVoice,
            int steps = DefaultSteps)
        {
            _root = string.IsNullOrWhiteSpace(streamingAssetsRoot) ? DefaultRoot : streamingAssetsRoot;
            _voice = string.IsNullOrWhiteSpace(voice) ? DefaultVoice : voice;
            Steps = Mathf.Clamp(steps, 1, 64);

            if (audioSource != null)
            {
                _audioSource = audioSource;
                return;
            }

            var host = new GameObject("LiteRtLmSupertonicAudio") { hideFlags = HideFlags.HideAndDontSave };
            UnityEngine.Object.DontDestroyOnLoad(host);
            _audioSource = host.AddComponent<AudioSource>();
            _audioSource.playOnAwake = false;
            _ownsAudioSource = true;
        }

        public string BackendName => "Supertonic (LiteRT)";

        /// <summary>Flow-matching steps. Fewer is faster and slightly less natural.</summary>
        public int Steps { get; set; }

        /// <summary>Seed for the latent noise. Fixing it makes a run reproducible against the bench.</summary>
        public int Seed { get; set; } = 1234;

        public bool IsAvailable
        {
            get
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                return EnsureClient() != null && Directory.Exists(RootPath);
#else
                return false;
#endif
            }
        }

        public bool IsSpeaking => _audioSource != null && _audioSource.isPlaying;

        public string LastError => _lastError;

        /// <summary>Where the last synthesis wrote its WAV; empty when nothing has run.</summary>
        public string LastWavPath { get; private set; } = string.Empty;

        /// <summary>JSON the bridge returned for the last call — timings, bucket, errors.</summary>
        public string LastResultJson { get; private set; } = string.Empty;

        private string RootPath => Path.Combine(Application.streamingAssetsPath, _root);

        public IEnumerator Speak(string text, string language, Action<LiteRtLmTtsResult> onComplete)
        {
            var result = new LiteRtLmTtsResult { Backend = BackendName };
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(text))
            {
                result.Error = "empty text";
                onComplete?.Invoke(result);
                yield break;
            }

            var startedAt = Time.realtimeSinceStartup;
            string wavPath = null;
            string json = null;
            string failure = null;

            try
            {
                json = Synthesize(text, language, out wavPath);
            }
            catch (Exception exception)
            {
                failure = exception.Message;
            }

            if (failure != null)
            {
                result.Error = failure;
                _lastError = failure;
                onComplete?.Invoke(result);
                yield break;
            }

            LastResultJson = json ?? string.Empty;
            var reportedError = ExtractJsonString(LastResultJson, "error");
            if (!string.IsNullOrEmpty(reportedError))
            {
                result.Error = reportedError;
                _lastError = reportedError;
                onComplete?.Invoke(result);
                yield break;
            }

            if (string.IsNullOrEmpty(wavPath) || !File.Exists(wavPath))
            {
                result.Error = "synthesis reported success but wrote no WAV";
                _lastError = result.Error;
                onComplete?.Invoke(result);
                yield break;
            }

            LastWavPath = wavPath;
            result.WavPath = wavPath;

            AudioClip clip = null;
            using (var request = UnityWebRequestMultimedia.GetAudioClip("file://" + wavPath, AudioType.WAV))
            {
                yield return request.SendWebRequest();
#if UNITY_2020_1_OR_NEWER
                var failed = request.result != UnityWebRequest.Result.Success;
#else
                var failed = request.isNetworkError || request.isHttpError;
#endif
                if (failed)
                {
                    result.Error = "could not load synthesized WAV: " + request.error;
                    _lastError = result.Error;
                    onComplete?.Invoke(result);
                    yield break;
                }

                clip = DownloadHandlerAudioClip.GetContent(request);
            }

            if (clip == null)
            {
                result.Error = "synthesized WAV decoded to no audio";
                _lastError = result.Error;
                onComplete?.Invoke(result);
                yield break;
            }

            _audioSource.clip = clip;
            _audioSource.Play();
            while (_audioSource.isPlaying)
            {
                yield return null;
            }

            result.Success = true;
            result.Seconds = Time.realtimeSinceStartup - startedAt;
            _lastError = string.Empty;
            onComplete?.Invoke(result);
#else
            result.Error =
                "Supertonic on LiteRT runs through the Android bridge; " +
                "use LiteRtLmSystemTts on the desktop.";
            onComplete?.Invoke(result);
            yield break;
#endif
        }

        public void Stop()
        {
            if (_audioSource != null && _audioSource.isPlaying)
            {
                _audioSource.Stop();
            }
        }

        public void Dispose()
        {
            Stop();

            if (_ownsAudioSource && _audioSource != null)
            {
                UnityEngine.Object.Destroy(_audioSource.gameObject);
            }

            _audioSource = null;
            _client = null;
            _text = null;
        }

        /// <summary>
        /// Smallest converted bucket that fits both lengths, or -1 when none does.
        /// The latent length is roughly 0.9x the id count in practice, so the ladder
        /// pairs them; the native side re-checks against the graph it actually
        /// loaded and reports a clear error rather than producing garbage.
        /// </summary>
        public static int ChooseBucket(int textIdCount, int estimatedLatentLength = 0)
        {
            var need = Mathf.Max(textIdCount, estimatedLatentLength);
            foreach (var bucket in Buckets)
            {
                if (need <= bucket)
                {
                    return bucket;
                }
            }

            return -1;
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private string Synthesize(string text, string language, out string wavPath)
        {
            wavPath = null;
            var client = EnsureClient();
            if (client == null)
            {
                throw new InvalidOperationException("Android bridge unavailable.");
            }

            _text ??= LiteRtLmSupertonicText.FromIndexerJson(
                File.ReadAllText(AssetPath("unicode_indexer.json"), Encoding.UTF8));

            _text.Encode(text, language, out var ids, out _);
            var bucket = ChooseBucket(ids.Length, ids.Length);
            if (bucket < 0)
            {
                throw new InvalidOperationException(
                    $"text needs {ids.Length} ids; the largest converted bucket is {Buckets[Buckets.Length - 1]}.");
            }

            var padded = new int[bucket];
            Array.Copy(ids, padded, ids.Length);

            wavPath = Path.Combine(
                Application.temporaryCachePath,
                "supertonic-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + ".wav");

            return client.RunSupertonicTts(
                GraphPath("dynamic", "duration_predictor"),
                GraphPath("dynamic", "text_encoder"),
                GraphPath("st-b" + bucket, "vector_estimator"),
                GraphPath("st-b" + bucket, "vocoder"),
                AssetPath("tts.json"),
                AssetPath(_voice + ".json"),
                padded,
                ids.Length,
                Steps,
                1.05f,
                "CPU",
                Seed,
                wavPath);
        }

        private LiteRtLmUnityClient EnsureClient()
        {
            _client ??= new LiteRtLmUnityClient();
            return _client.IsAvailable ? _client : null;
        }
#else
        private LiteRtLmUnityClient EnsureClient() => null;
#endif

        private string AssetPath(string name) => Path.Combine(RootPath, "assets", name);

        /// <summary>
        /// The converted file name carries its precision (`_float32`, `_w8`), which
        /// is a deployment choice rather than something callers should encode, so
        /// the directory is scanned instead of the name being constructed.
        /// </summary>
        private string GraphPath(string bucketDir, string stem)
        {
            var directory = Path.Combine(RootPath, bucketDir, stem);
            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException(directory);
            }

            var files = Directory.GetFiles(directory, "*.tflite");
            if (files.Length == 0)
            {
                throw new FileNotFoundException("no .tflite under " + directory);
            }

            Array.Sort(files, StringComparer.Ordinal);
            return files[0];
        }

        /// <summary>
        /// Minimal string field reader — enough to surface a bridge error without
        /// pulling a JSON dependency into the runtime assembly.
        /// </summary>
        internal static string ExtractJsonString(string json, string key)
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

            var builder = new StringBuilder();
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

        /// <summary>Reads a numeric field from the bridge JSON; NaN when absent.</summary>
        public static double ExtractJsonNumber(string json, string key)
        {
            if (string.IsNullOrEmpty(json))
            {
                return double.NaN;
            }

            var needle = "\"" + key + "\"";
            var at = json.IndexOf(needle, StringComparison.Ordinal);
            if (at < 0)
            {
                return double.NaN;
            }

            var colon = json.IndexOf(':', at + needle.Length);
            if (colon < 0)
            {
                return double.NaN;
            }

            var end = colon + 1;
            while (end < json.Length && (char.IsWhiteSpace(json[end]) || json[end] == '+'))
            {
                end++;
            }

            var start = end;
            while (end < json.Length &&
                   (char.IsDigit(json[end]) || json[end] == '.' || json[end] == '-' ||
                    json[end] == 'e' || json[end] == 'E'))
            {
                end++;
            }

            return double.TryParse(json.Substring(start, end - start),
                                   NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
                ? value
                : double.NaN;
        }
    }
}
