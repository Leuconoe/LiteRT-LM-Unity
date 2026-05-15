using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmAsrSmokeTestRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM ASRSmoke]";
        private const string StatusFileName = "LiteRtLmAsrSmokeTest.status.txt";

        [SerializeField] private string modelPath = "parakeet_tdt_0.6b_v3_5s_i8.tflite";
        [SerializeField] private string audioPath = "Tactical Evaluation Results Report - March 5, 2025.mp3";
        [SerializeField] private string tokenizerJsonPath = "parakeet-tdt-0.6b-v3/tokenizer.json";
        [SerializeField] private string backend = "GPU_FP16";

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
            WriteStatus("START", $"backend={backend}, model={modelPath}, audio={audioPath}, platform={Application.platform}");

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

                var startedAt = Time.realtimeSinceStartup;
                var inspectionJson = client.InspectLiteRtModel(resolvedModelPath);
                WriteStatus("INSPECT_RESULT", OneLine(Truncate(inspectionJson, 3000)));

                WriteStatus("ASR_INVOKE", $"Invoking Parakeet LiteRT encode/decode smoke path with backend={backend}.");
                var asrJson = client.RunParakeetAsrSmoke(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath, backend);
                var elapsedSeconds = Time.realtimeSinceStartup - startedAt;

                if (string.IsNullOrWhiteSpace(asrJson))
                {
                    throw new InvalidOperationException("Parakeet ASR smoke test returned an empty result.");
                }
                if (asrJson.Contains("\"success\": false", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidOperationException($"Parakeet ASR smoke test reported failure: {OneLine(Truncate(asrJson, 1000))}");
                }

                WriteStatus("ASR_RESULT", OneLine(Truncate(asrJson, 3000)));
                WriteStatus("SUCCESS", $"elapsedSeconds={elapsedSeconds:0.###}, resultLength={asrJson.Length}");
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
    }
}
