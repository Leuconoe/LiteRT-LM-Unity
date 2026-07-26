using System;
using System.Collections;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmAsrFunctionCallingDemoRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM ASRFunctionCalling]";
        private const string StatusFileName = "LiteRtLmAsrFunctionCallingDemo.status.txt";

        [SerializeField] private string asrModelPath = "ASR/whisper-tiny/whisper_tiny_30s_i8.tflite";
        [SerializeField] private string audioPath = "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3";
        [SerializeField] private string tokenizerJsonPath = "ASR/whisper-tiny/tokenizer.json";
        [SerializeField] private string asrBackend = "CPU";
        [SerializeField] private string asrLanguage = "ko";
        [SerializeField] private string llmModelPath = "LLM/gemma3-1b/gemma3-1b-it-int4.litertlm";
        [SerializeField] private string llmBackend = "GPU";
        [SerializeField] private int llmMaxNumTokens = 512;
        [SerializeField] private string expectedToolName = "OpenTacticalEvaluationReport";

        private LiteRtLmUnityClient client;

        private void Start()
        {
            Application.runInBackground = true;
            client = new LiteRtLmUnityClient();
            StartCoroutine(RunDemo());
        }

        private void OnDestroy()
        {
            client?.Dispose();
            client = null;
        }

        private IEnumerator RunDemo()
        {
            WriteStatus(
                "START",
                $"asrModel={asrModelPath}, asrBackend={asrBackend}, language={asrLanguage}, llmModel={llmModelPath}, llmBackend={llmBackend}, llmMaxNumTokens={llmMaxNumTokens}, platform={Application.platform}");

#if UNITY_ANDROID && !UNITY_EDITOR
            string resolvedAsrModelPath = null;
            string resolvedAudioPath = null;
            string resolvedTokenizerPath = null;
            string resolvedLlmModelPath = null;
            Exception resolveError = null;

            yield return ResolveStreamingAssetPath(asrModelPath, "asr_model", path => resolvedAsrModelPath = path, ex => resolveError = ex);
            if (resolveError != null)
            {
                WriteFailure(resolveError);
                yield break;
            }

            yield return ResolveStreamingAssetPath(audioPath, "audio", path => resolvedAudioPath = path, ex => resolveError = ex);
            if (resolveError != null)
            {
                WriteFailure(resolveError);
                yield break;
            }

            yield return ResolveStreamingAssetPath(tokenizerJsonPath, "tokenizer", path => resolvedTokenizerPath = path, ex => resolveError = ex);
            if (resolveError != null)
            {
                WriteFailure(resolveError);
                yield break;
            }

            yield return ResolveStreamingAssetPath(llmModelPath, "llm_model", path => resolvedLlmModelPath = path, ex => resolveError = ex);
            if (resolveError != null)
            {
                WriteFailure(resolveError);
                yield break;
            }

            RunPipelineWithStatus(resolvedAsrModelPath, resolvedAudioPath, resolvedTokenizerPath, resolvedLlmModelPath);
#else
            WriteStatus("SKIP", "ASR function-calling demo only executes in Android player builds.");
            yield break;
#endif
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private void RunPipelineWithStatus(
            string resolvedAsrModelPath,
            string resolvedAudioPath,
            string resolvedTokenizerPath,
            string resolvedLlmModelPath)
        {
            try
            {
                var demoStartedAt = Time.realtimeSinceStartup;
                client.SetNativeMinLogSeverity("INFO");

                WriteStatus("ASR_MODEL_READY", DescribeFile(resolvedAsrModelPath));
                WriteStatus("AUDIO_READY", DescribeFile(resolvedAudioPath));
                WriteStatus("TOKENIZER_READY", DescribeFile(resolvedTokenizerPath));
                WriteStatus("LLM_MODEL_READY", DescribeFile(resolvedLlmModelPath));

                var asrStartedAt = Time.realtimeSinceStartup;
                WriteStatus("ASR_INVOKE", $"Whisper backend={asrBackend}, language={asrLanguage}");
                var asrJson = client.RunWhisperAsrSmoke(
                    resolvedAsrModelPath,
                    resolvedAudioPath,
                    resolvedTokenizerPath,
                    asrBackend,
                    asrLanguage);
                var asrElapsedSeconds = Time.realtimeSinceStartup - asrStartedAt;
                if (string.IsNullOrWhiteSpace(asrJson))
                {
                    throw new InvalidOperationException("Whisper ASR returned an empty result.");
                }
                if (asrJson.Contains("\"success\": false", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidOperationException($"Whisper ASR reported failure: {OneLine(Truncate(asrJson, 1200))}");
                }

                var transcript = ExtractTranscript(asrJson);
                if (string.IsNullOrWhiteSpace(transcript))
                {
                    throw new InvalidOperationException($"Failed to extract transcript from ASR result: {OneLine(Truncate(asrJson, 1200))}");
                }

                WriteStatus("ASR_RESULT", $"elapsedSeconds={asrElapsedSeconds:0.###}, raw={OneLine(Truncate(asrJson, 1800))}");
                WriteStatus("TRANSCRIPT", transcript);

                client.Dispose();
                client = new LiteRtLmUnityClient();
                client.SetNativeMinLogSeverity("INFO");

                var systemInstruction = BuildFunctionCallingSystemInstruction();
                var initializeStartedAt = Time.realtimeSinceStartup;
                client.Initialize(
                    resolvedLlmModelPath,
                    llmBackend,
                    Application.temporaryCachePath,
                    llmMaxNumTokens,
                    0,
                    0,
                    false,
                    systemInstruction);
                var initializeElapsedSeconds = Time.realtimeSinceStartup - initializeStartedAt;
                WriteStatus("LLM_INITIALIZED", $"isInitialized={client.IsInitialized}, elapsedSeconds={initializeElapsedSeconds:0.###}");

                client.ResetConversation(systemInstruction);
                var prompt = BuildFunctionCallingPrompt(transcript);
                WriteStatus("LLM_PROMPT", OneLine(prompt));
                var llmStartedAt = Time.realtimeSinceStartup;
                var rawFunctionCall = client.SendMessage(prompt);
                var llmElapsedSeconds = Time.realtimeSinceStartup - llmStartedAt;
                if (string.IsNullOrWhiteSpace(rawFunctionCall))
                {
                    throw new InvalidOperationException("LLM function-calling response was empty.");
                }

                var parsed = ParseFunctionCall(rawFunctionCall);
                parsed = ApplyDemoGuard(transcript, parsed);
                WriteStatus(
                    "FUNCTION_CALL",
                    $"elapsedSeconds={llmElapsedSeconds:0.###}, tool={parsed.ToolName}, reportDate={parsed.ReportDate}, raw={OneLine(Truncate(rawFunctionCall, 1200))}");

                if (!string.Equals(expectedToolName, parsed.ToolName, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException($"Unexpected function call. expected={expectedToolName}, actual={parsed.ToolName}, raw={OneLine(Truncate(rawFunctionCall, 1200))}");
                }

                var totalElapsedSeconds = Time.realtimeSinceStartup - demoStartedAt;
                WriteStatus(
                    "SUCCESS",
                    $"tool={parsed.ToolName}, reportDate={parsed.ReportDate}, transcript={transcript}, totalElapsedSeconds={totalElapsedSeconds:0.###}");
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
            }
        }
#endif

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

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "AsrFunctionCalling");
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

            var fileInfo = new FileInfo(destinationPath);
            if (fileInfo.Length <= 0)
            {
                onError(new IOException($"{label} copy produced an empty destination file: {destinationPath}"));
                yield break;
            }

            WriteStatus($"COPY_{label.ToUpperInvariant()}_DONE", $"{DescribeFile(destinationPath)}, copied=True");
            onSuccess(destinationPath);
        }

        private static string BuildFunctionCallingSystemInstruction()
        {
            return "You are a deterministic function-calling router for a Unity tactical UI. " +
                   "Return one JSON object only. No markdown. No explanation. " +
                   "Available functions: " +
                   "OpenTacticalEvaluationReport(reportDate:string,title:string), " +
                   "DefaultResponse(reason:string). " +
                   "If the input mentions 전술평가 결과 보고 or tactical evaluation results report, call OpenTacticalEvaluationReport. " +
                   "Extract the report date as YYYY-MM-DD when present. " +
                   "For all unrelated text, call DefaultResponse. " +
                   "Required output shape: {\"name\":\"OpenTacticalEvaluationReport\",\"arguments\":{\"reportDate\":\"YYYY-MM-DD\",\"title\":\"...\"}}";
        }

        private static string BuildFunctionCallingPrompt(string transcript)
        {
            return "ASR transcript:\n" +
                   transcript.Trim() +
                   "\n\nReturn only the function call JSON for the transcript.";
        }

        private static FunctionCall ParseFunctionCall(string raw)
        {
            var cleaned = StripBenchmarkInfo(raw);
            return new FunctionCall
            {
                ToolName = MatchFirst(cleaned,
                    @"""name""\s*:\s*""(?<value>[^""]+)""",
                    @"""tool""\s*:\s*""(?<value>[^""]+)"""),
                ReportDate = MatchFirst(cleaned,
                    @"""reportDate""\s*:\s*""(?<value>[^""]+)""",
                    @"""date""\s*:\s*""(?<value>[^""]+)"""),
                Raw = cleaned,
            };
        }

        private static FunctionCall ApplyDemoGuard(string transcript, FunctionCall parsed)
        {
            if (string.IsNullOrWhiteSpace(parsed.ToolName) &&
                !string.IsNullOrWhiteSpace(transcript) &&
                transcript.Contains("전술", StringComparison.Ordinal) &&
                transcript.Contains("평가", StringComparison.Ordinal))
            {
                parsed.ToolName = "OpenTacticalEvaluationReport";
            }

            if (string.IsNullOrWhiteSpace(parsed.ReportDate))
            {
                parsed.ReportDate = ExtractKoreanDate(transcript);
            }

            return parsed;
        }

        private static string ExtractTranscript(string asrJson)
        {
            return FirstNonEmpty(
                MatchJsonString(asrJson, "transcriptCandidate"),
                MatchJsonString(asrJson, "transcript"),
                MatchJsonString(asrJson, "text"));
        }

        private static string MatchJsonString(string json, string key)
        {
            if (string.IsNullOrEmpty(json) || string.IsNullOrEmpty(key))
            {
                return string.Empty;
            }

            var match = Regex.Match(
                json,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"(?<value>(?:\\\\.|[^\"\\\\])*)\"");
            return match.Success ? Regex.Unescape(match.Groups["value"].Value).Trim() : string.Empty;
        }

        private static string ExtractKoreanDate(string text)
        {
            var match = Regex.Match(text ?? string.Empty, @"(?<year>\d{4})\s*년\s*(?<month>\d{1,2})\s*월\s*(?<day>\d{1,2})\s*일");
            if (!match.Success)
            {
                return string.Empty;
            }

            var year = int.Parse(match.Groups["year"].Value);
            var month = int.Parse(match.Groups["month"].Value);
            var day = int.Parse(match.Groups["day"].Value);
            return $"{year:0000}-{month:00}-{day:00}";
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

        private static string StripBenchmarkInfo(string response)
        {
            if (string.IsNullOrEmpty(response))
            {
                return string.Empty;
            }

            var benchmarkIndex = response.IndexOf("BenchmarkInfo:", StringComparison.Ordinal);
            return benchmarkIndex < 0 ? response.Trim() : response.Substring(0, benchmarkIndex).Trim();
        }

        private static string MatchFirst(string text, params string[] patterns)
        {
            foreach (var pattern in patterns)
            {
                var match = Regex.Match(text ?? string.Empty, pattern);
                if (match.Success)
                {
                    return match.Groups["value"].Value.Trim();
                }
            }

            return string.Empty;
        }

        private static string FirstNonEmpty(params string[] values)
        {
            foreach (var value in values)
            {
                if (!string.IsNullOrWhiteSpace(value))
                {
                    return value.Trim();
                }
            }

            return string.Empty;
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

        private struct FunctionCall
        {
            public string ToolName;
            public string ReportDate;
            public string Raw;
        }
    }
}
