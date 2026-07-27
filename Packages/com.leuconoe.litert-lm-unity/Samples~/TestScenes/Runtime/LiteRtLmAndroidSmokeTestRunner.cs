using System;
using System.Collections;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmAndroidSmokeTestRunner : MonoBehaviour, ILiteRtLmModelHost
    {
        private const string LogPrefix = "[LiteRT-LM AndroidSmoke]";
        private const string StatusFileName = "LiteRtLmAndroidSmokeTest.status.txt";
        private const string ConfigFileName = "LiteRtLmAndroidSmokeTest.config.json";

        [SerializeField] private string modelPath = "model.litertlm";
        [SerializeField] private string backend = "GPU";
        [SerializeField] private int maxNumTokens = 64;
        [SerializeField] private int maxNumImages;
        [SerializeField] private bool enableSpeculativeDecoding;
        [SerializeField] private bool runStandaloneBenchmark;
        [SerializeField] private int benchmarkPrefillTokens = 64;
        [SerializeField] private int benchmarkDecodeTokens = 32;
        [SerializeField] private int benchmarkRuns = 3;
        [SerializeField] private string systemInstruction = "You are a concise Unity Android smoke test assistant.";
        [SerializeField] private bool resetConversationBeforeEachPrompt = true;

        // Optional multimodal smoke turns (runtime-config driven). Paths must be
        // absolute device paths (e.g. pushed via adb) or resolvable files.
        // visionBackend/audioBackend must be set (e.g. "CPU") for the engine to
        // load the vision/audio executors; empty string leaves them disabled.
        [SerializeField] private string visionBackend = "";
        [SerializeField] private string audioBackend = "";
        [SerializeField] private bool skipTextTurns;
        [SerializeField] private string mediaImagePath = "";
        [SerializeField] private string mediaImagePrompt = "Describe this image briefly.";
        [SerializeField] private string mediaAudioPath = "";
        [SerializeField] private string mediaAudioPrompt = "Transcribe the audio:";

        private readonly string[] prompts =
        {
            "Reply with exactly: Android LiteRT-LM smoke test turn one.",
            "Reply with exactly: Android LiteRT-LM smoke test turn two.",
        };

        private LiteRtLmUnityClient client;

        private void Start()
        {
            Application.runInBackground = true;
            client = new LiteRtLmUnityClient();
            StartCoroutine(RunSmokeTest());
        }

        private void OnDestroy()
        {
            ReleaseModels();
        }

        /// <inheritdoc />
        public void ReleaseModels()
        {
            client?.Dispose();
            client = null;
        }

        private IEnumerator RunSmokeTest()
        {
            ApplyRuntimeConfigOverrides();
            WriteStatus("START", $"backend={backend}, model={modelPath}, platform={Application.platform}, speculativeDecoding={enableSpeculativeDecoding}, maxNumTokens={maxNumTokens}, maxNumImages={maxNumImages}, benchmarkPrefillTokens={benchmarkPrefillTokens}, benchmarkDecodeTokens={benchmarkDecodeTokens}, benchmarkRuns={benchmarkRuns}, resetConversationBeforeEachPrompt={resetConversationBeforeEachPrompt}");

#if UNITY_ANDROID && !UNITY_EDITOR
            var resolvedModelPath = string.Empty;
            Exception resolveError = null;
            yield return ResolveModelPath(
                modelPath,
                path => resolvedModelPath = path,
                ex => resolveError = ex);

            if (resolveError != null)
            {
                WriteFailure(resolveError);
                yield break;
            }

            RunInferenceWithStatus(resolvedModelPath);
#else
            WriteStatus("SKIP", "Android smoke test runner only executes in Android player builds.");
            yield break;
#endif
        }

        private void RunInferenceWithStatus(string resolvedModelPath)
        {
            try
            {
                var testStartedAt = Time.realtimeSinceStartup;
                client.SetNativeMinLogSeverity("INFO");
                WriteStatus("INITIALIZE", $"resolvedModel={resolvedModelPath}, cacheDir={Application.temporaryCachePath}, maxNumTokens={maxNumTokens}, maxNumImages={maxNumImages}, speculativeDecoding={enableSpeculativeDecoding}");
                var initializeStartedAt = Time.realtimeSinceStartup;
                client.Initialize(
                    resolvedModelPath,
                    backend,
                    Application.temporaryCachePath,
                    maxNumTokens,
                    maxNumImages,
                    0,
                    enableSpeculativeDecoding,
                    systemInstruction,
                    visionBackend: visionBackend,
                    audioBackend: audioBackend);
                var initializeElapsedSeconds = Time.realtimeSinceStartup - initializeStartedAt;
                WriteStatus("INITIALIZED", $"isInitialized={client.IsInitialized}, elapsedSeconds={initializeElapsedSeconds:0.###}");

                if (skipTextTurns)
                {
                    WriteStatus("TEXT_TURNS_SKIPPED", "skipTextTurns=true (media-only smoke run).");
                }
                else
                {
                    RunPromptTurns();
                }

                RunMediaTurns();

                if (runStandaloneBenchmark && benchmarkPrefillTokens > 0 && benchmarkDecodeTokens > 0)
                {
                    RunStandaloneBenchmark(resolvedModelPath);
                }
                else
                {
                    WriteStatus(
                        "BENCHMARK_SKIPPED",
                        "Standalone benchmark disabled to avoid creating a second LiteRT-LM engine during smoke tests.");
                }

                var totalElapsedSeconds = Time.realtimeSinceStartup - testStartedAt;
                WriteStatus("SUCCESS", $"backend={backend}, turns={prompts.Length}, totalElapsedSeconds={totalElapsedSeconds:0.###}");
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
            }
        }

        private void RunPromptTurns()
        {
            for (var i = 0; i < prompts.Length; i++)
            {
                var turn = i + 1;
                var prompt = prompts[i];
                if (resetConversationBeforeEachPrompt)
                {
                    WriteStatus("RESET_CONVERSATION", $"{turn}/{prompts.Length}: clearing previous conversation state before prompt.");
                    client.ResetConversation(systemInstruction);
                }

                WriteStatus("TURN", $"{turn}/{prompts.Length}: prompt={prompt}");
                var startedAt = Time.realtimeSinceStartup;
                var response = client.SendMessage(prompt);
                var elapsedSeconds = Time.realtimeSinceStartup - startedAt;

                if (string.IsNullOrWhiteSpace(response))
                {
                    throw new InvalidOperationException($"Turn {turn} returned an empty response.");
                }

                WriteStatus(
                    "RESPONSE",
                    $"{turn}/{prompts.Length}: elapsedSeconds={elapsedSeconds:0.###}, length={response.Length}, preview={OneLine(Truncate(response, 180))}");
            }
        }

        private void RunMediaTurns()
        {
            if (!string.IsNullOrWhiteSpace(mediaImagePath))
            {
                RunMediaTurn("image", mediaImagePrompt, mediaImagePath, string.Empty);
            }

            if (!string.IsNullOrWhiteSpace(mediaAudioPath))
            {
                RunMediaTurn("audio", mediaAudioPrompt, string.Empty, mediaAudioPath);
            }
        }

        private void RunMediaTurn(string label, string prompt, string imagePath, string audioPath)
        {
            var mediaPath = string.IsNullOrWhiteSpace(imagePath) ? audioPath : imagePath;
            if (!File.Exists(mediaPath))
            {
                throw new FileNotFoundException($"Media {label} file not found: {mediaPath}", mediaPath);
            }

            if (resetConversationBeforeEachPrompt)
            {
                WriteStatus("RESET_CONVERSATION", $"media-{label}: clearing previous conversation state before prompt.");
                client.ResetConversation(systemInstruction);
            }

            WriteStatus("MEDIA_TURN", $"{label}: prompt={prompt}, path={mediaPath}, bytes={new FileInfo(mediaPath).Length}");
            var startedAt = Time.realtimeSinceStartup;
            var response = client.SendMessageWithMedia(prompt, null, imagePath, audioPath);
            var elapsedSeconds = Time.realtimeSinceStartup - startedAt;

            if (string.IsNullOrWhiteSpace(response))
            {
                throw new InvalidOperationException($"Media {label} turn returned an empty response.");
            }

            WriteStatus(
                "MEDIA_RESPONSE",
                $"{label}: elapsedSeconds={elapsedSeconds:0.###}, length={response.Length}, preview={OneLine(Truncate(response, 400))}");
        }

        private void RunStandaloneBenchmark(string resolvedModelPath)
        {
            client.Dispose();
            client = null;

            var runs = Math.Max(1, benchmarkRuns);
            var elapsedTotalSeconds = 0f;
            WriteStatus("BENCHMARK", $"runs={runs}, prefillTokens={benchmarkPrefillTokens}, decodeTokens={benchmarkDecodeTokens}");
            for (var run = 1; run <= runs; run++)
            {
                using var benchmarkClient = new LiteRtLmUnityClient();
                benchmarkClient.SetNativeMinLogSeverity("INFO");
                var benchmarkStartedAt = Time.realtimeSinceStartup;
                var benchmark = benchmarkClient.RunBenchmark(
                    resolvedModelPath,
                    backend,
                    Application.temporaryCachePath,
                    benchmarkPrefillTokens,
                    benchmarkDecodeTokens,
                    enableSpeculativeDecoding);
                var benchmarkElapsedSeconds = Time.realtimeSinceStartup - benchmarkStartedAt;
                elapsedTotalSeconds += benchmarkElapsedSeconds;
                WriteStatus("BENCHMARK_RESULT", $"run={run}/{runs}, elapsedSeconds={benchmarkElapsedSeconds:0.###}, {OneLine(benchmark)}");
            }

            WriteStatus("BENCHMARK_SUMMARY", $"runs={runs}, averageElapsedSeconds={(elapsedTotalSeconds / runs):0.###}");
        }

        private IEnumerator ResolveModelPath(string configuredPath, Action<string> onSuccess, Action<Exception> onError)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
            {
                onError(new ArgumentException("Model path is required."));
                yield break;
            }

            if (Path.IsPathRooted(configuredPath))
            {
                onSuccess(configuredPath);
                yield break;
            }

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM");
            Directory.CreateDirectory(destinationDirectory);
            var destinationPath = Path.Combine(destinationDirectory, configuredPath);
            var destinationParent = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(destinationParent))
            {
                Directory.CreateDirectory(destinationParent);
            }

            if (File.Exists(destinationPath))
            {
                var existingFileInfo = new FileInfo(destinationPath);
                WriteStatus("MODEL_READY", $"path={destinationPath}, bytes={existingFileInfo.Length}, copied=False");
                onSuccess(destinationPath);
                yield break;
            }

            var sourcePath = Path.Combine(Application.streamingAssetsPath, configuredPath).Replace("\\", "/");
            WriteStatus("COPY_MODEL", $"source={sourcePath}, destination={destinationPath}");

            using var request = UnityWebRequest.Get(sourcePath);
            request.downloadHandler = new DownloadHandlerFile(destinationPath)
            {
                removeFileOnAbort = true,
            };

            yield return request.SendWebRequest();
            if (request.result != UnityWebRequest.Result.Success)
            {
                onError(new IOException($"Failed to copy model from StreamingAssets: {request.error}"));
                yield break;
            }

            if (!File.Exists(destinationPath))
            {
                onError(new FileNotFoundException($"Model copy did not create destination file: {destinationPath}", destinationPath));
                yield break;
            }

            var copiedFileInfo = new FileInfo(destinationPath);
            if (copiedFileInfo.Length <= 0)
            {
                onError(new IOException($"Model copy produced an empty destination file: {destinationPath}"));
                yield break;
            }

            WriteStatus("COPY_MODEL_DONE", $"destination={destinationPath}, bytes={copiedFileInfo.Length}");
            WriteStatus("MODEL_READY", $"path={destinationPath}, bytes={copiedFileInfo.Length}, copied=True");
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
                if (TryGetJsonString(json, "backend", out var configuredBackend))
                {
                    backend = configuredBackend;
                }
                if (TryGetJsonInt(json, "maxNumTokens", out var configuredMaxNumTokens) && configuredMaxNumTokens > 0)
                {
                    maxNumTokens = configuredMaxNumTokens;
                }
                if (TryGetJsonInt(json, "maxNumImages", out var configuredMaxNumImages) && configuredMaxNumImages >= 0)
                {
                    maxNumImages = configuredMaxNumImages;
                }
                if (TryGetJsonBool(json, "enableSpeculativeDecoding", out var configuredSpeculativeDecoding))
                {
                    enableSpeculativeDecoding = configuredSpeculativeDecoding;
                }
                if (TryGetJsonBool(json, "runStandaloneBenchmark", out var configuredRunStandaloneBenchmark))
                {
                    runStandaloneBenchmark = configuredRunStandaloneBenchmark;
                }
                if (TryGetJsonInt(json, "benchmarkPrefillTokens", out var configuredPrefillTokens) && configuredPrefillTokens > 0)
                {
                    benchmarkPrefillTokens = configuredPrefillTokens;
                }
                if (TryGetJsonInt(json, "benchmarkDecodeTokens", out var configuredDecodeTokens) && configuredDecodeTokens > 0)
                {
                    benchmarkDecodeTokens = configuredDecodeTokens;
                }
                if (TryGetJsonInt(json, "benchmarkRuns", out var configuredBenchmarkRuns) && configuredBenchmarkRuns > 0)
                {
                    benchmarkRuns = configuredBenchmarkRuns;
                }
                if (TryGetJsonString(json, "systemInstruction", out var configuredSystemInstruction))
                {
                    systemInstruction = configuredSystemInstruction;
                }
                if (TryGetJsonBool(json, "resetConversationBeforeEachPrompt", out var configuredResetConversation))
                {
                    resetConversationBeforeEachPrompt = configuredResetConversation;
                }
                if (TryGetJsonString(json, "visionBackend", out var configuredVisionBackend))
                {
                    visionBackend = configuredVisionBackend;
                }
                if (TryGetJsonString(json, "audioBackend", out var configuredAudioBackend))
                {
                    audioBackend = configuredAudioBackend;
                }
                if (TryGetJsonBool(json, "skipTextTurns", out var configuredSkipTextTurns))
                {
                    skipTextTurns = configuredSkipTextTurns;
                }
                if (TryGetJsonString(json, "mediaImagePath", out var configuredMediaImagePath))
                {
                    mediaImagePath = configuredMediaImagePath;
                }
                if (TryGetJsonString(json, "mediaImagePrompt", out var configuredMediaImagePrompt))
                {
                    mediaImagePrompt = configuredMediaImagePrompt;
                }
                if (TryGetJsonString(json, "mediaAudioPath", out var configuredMediaAudioPath))
                {
                    mediaAudioPath = configuredMediaAudioPath;
                }
                if (TryGetJsonString(json, "mediaAudioPrompt", out var configuredMediaAudioPrompt))
                {
                    mediaAudioPrompt = configuredMediaAudioPrompt;
                }

                WriteStatus("CONFIG", $"loaded={configPath}, model={modelPath}, backend={backend}, audioBackend={audioBackend}, skipTextTurns={skipTextTurns}, mediaImagePath={mediaImagePath}, mediaAudioPath={mediaAudioPath}");
            }
            catch (Exception ex)
            {
                WriteFailure(new InvalidOperationException($"Failed to load Android smoke runtime config: {configPath}", ex));
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
            if (match.Success && int.TryParse(match.Groups[1].Value, out value))
            {
                return true;
            }

            value = 0;
            return false;
        }

        private static bool TryGetJsonBool(string json, string propertyName, out bool value)
        {
            var match = Regex.Match(json, $"\"{Regex.Escape(propertyName)}\"\\s*:\\s*(true|false)", RegexOptions.IgnoreCase);
            if (match.Success && bool.TryParse(match.Groups[1].Value, out value))
            {
                return true;
            }

            value = false;
            return false;
        }
    }
}
