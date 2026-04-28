using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmAndroidSmokeTestRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM AndroidSmoke]";
        private const string StatusFileName = "LiteRtLmAndroidSmokeTest.status.txt";

        [SerializeField] private string modelPath = "model.litertlm";
        [SerializeField] private string backend = "GPU";
        [SerializeField] private int maxNumTokens = 64;
        [SerializeField] private bool runStandaloneBenchmark;
        [SerializeField] private int benchmarkPrefillTokens = 64;
        [SerializeField] private int benchmarkDecodeTokens = 32;
        [SerializeField] private string systemInstruction = "You are a concise Unity Android smoke test assistant.";
        [SerializeField] private bool resetConversationBeforeEachPrompt = true;

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
            client?.Dispose();
            client = null;
        }

        private IEnumerator RunSmokeTest()
        {
            WriteStatus("START", $"backend={backend}, model={modelPath}, platform={Application.platform}, resetConversationBeforeEachPrompt={resetConversationBeforeEachPrompt}");

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
                WriteStatus("INITIALIZE", $"resolvedModel={resolvedModelPath}, cacheDir={Application.temporaryCachePath}");
                var initializeStartedAt = Time.realtimeSinceStartup;
                client.Initialize(
                    resolvedModelPath,
                    backend,
                    Application.temporaryCachePath,
                    maxNumTokens,
                    0,
                    0,
                    systemInstruction);
                var initializeElapsedSeconds = Time.realtimeSinceStartup - initializeStartedAt;
                WriteStatus("INITIALIZED", $"isInitialized={client.IsInitialized}, elapsedSeconds={initializeElapsedSeconds:0.###}");

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

                if (runStandaloneBenchmark && benchmarkPrefillTokens > 0 && benchmarkDecodeTokens > 0)
                {
                    WriteStatus("BENCHMARK", $"prefillTokens={benchmarkPrefillTokens}, decodeTokens={benchmarkDecodeTokens}");
                    var benchmark = client.RunBenchmark(
                        resolvedModelPath,
                        backend,
                        Application.temporaryCachePath,
                        benchmarkPrefillTokens,
                        benchmarkDecodeTokens);
                    WriteStatus("BENCHMARK_RESULT", benchmark);
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
    }
}
