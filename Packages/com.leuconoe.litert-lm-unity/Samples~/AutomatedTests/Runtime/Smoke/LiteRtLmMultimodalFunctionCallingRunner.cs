using System;
using System.Collections;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Multimodal function-calling demo pipeline.
    /// On Android it runs the real multimodal stage 1: the configured media
    /// (image and/or audio) plus the user utterance are sent through
    /// SendMessageWithMedia to a media-capable model (e.g. gemma-4 E2B) with
    /// the tools JSON embedded in the prompt (text-embedded tools, mirroring
    /// the Windows degraded-mode tools contract), and the returned tool call
    /// is validated against expectedToolName. Runtime overrides come from
    /// LiteRtLmMultimodalFunctionCallingDemo.config.json in persistentDataPath.
    /// On Windows Editor/Player it runs a text-degraded mode: the configured
    /// fallback utterance stands in for the multimodal transcription and is routed
    /// through the Windows CLI with constrained function calling.
    /// </summary>
    public sealed class LiteRtLmMultimodalFunctionCallingRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM MultimodalFunctionCalling]";
        private const string StatusFileName = "LiteRtLmMultimodalFunctionCallingDemo.status.txt";
        private const string ConfigFileName = "LiteRtLmMultimodalFunctionCallingDemo.config.json";
        private const string ReferenceNow = "2026-04-24 10:30:00";

        // take3+ AAR exposes sendMessageWithMedia(text, imageBytes, imagePath, audioPath, extraContextJson).
        private const bool MediaApiAvailable = true;

        [Header("Multimodal Input")]
        [SerializeField] private Texture2D testImage;
        [SerializeField] private string audioPath = "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3";

        [Header("Android On-Device (runtime-config overridable)")]
        // Absolute device path to a media-capable .litertlm model (e.g. gemma-4 E2B).
        [SerializeField] private string androidLlmModelPath = "";
        [SerializeField] private string androidBackend = "CPU";
        // visionBackend/audioBackend must be non-empty for the engine to load the executors.
        [SerializeField] private string androidVisionBackend = "CPU";
        [SerializeField] private string androidAudioBackend = "";
        // Image turns overflow the KV cache below ~4000 tokens (validated gemma-4 config).
        [SerializeField] private int androidMaxNumTokens = 4000;
        [SerializeField] private int androidMaxNumImages = 1;
        // Absolute device path to the stage-1 image; falls back to testImage bytes when empty.
        [SerializeField] private string mediaImagePath = "";
        // Absolute device path to optional stage-1 audio.
        [SerializeField] private string mediaAudioPath = "";

        [Header("Text-Degraded Fallback (Windows Editor)")]
        [SerializeField] private string textFallbackUtterance = "멀티모달 데이터 목록을 화면에 띄워줘.";
        [SerializeField] private string expectedToolName = "ShowMultimodalDataList";

        [Header("LLM")]
        [SerializeField] private string llmModelPath = "LLM/gemma3-1b/gemma3-1b-it-int4.litertlm";
        [SerializeField] private string windowsCliExecutablePath = "Tools/Windows/Bin/litert_lm_main.windows_x86_64.exe";
        [SerializeField] private string windowsBackend = "CPU";
        [SerializeField] private float timeoutSeconds = 120f;
        [SerializeField] private bool enableConstrainedDecoding = true;
        [SerializeField] private bool outputMessageJson = true;

        private void Start()
        {
            Application.runInBackground = true;
            StartCoroutine(RunDemo());
        }

        private IEnumerator RunDemo()
        {
            WriteStatus(
                "START",
                $"llmModel={llmModelPath}, windowsBackend={windowsBackend}, expectedTool={expectedToolName}, " +
                $"image={(testImage == null ? "(none)" : testImage.name)}, audio={audioPath}, " +
                $"mediaApiAvailable={MediaApiAvailable}, platform={Application.platform}");

#if UNITY_ANDROID && !UNITY_EDITOR
            yield return RunAndroidMultimodalPipeline();
#else
            if (Application.platform != RuntimePlatform.WindowsEditor &&
                Application.platform != RuntimePlatform.WindowsPlayer)
            {
                WriteStatus("SKIP", "Multimodal function-calling demo requires Android (pending bridge API) or Windows CLI text-degraded mode.");
                yield break;
            }

            yield return RunWindowsTextDegradedPipeline();
#endif
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private IEnumerator RunAndroidMultimodalPipeline()
        {
            var demoStartedAt = Time.realtimeSinceStartup;
            LoadAndroidRuntimeConfig();

            if (!MediaApiAvailable)
            {
                WriteStatus("PENDING_BRIDGE_API", "bridge API pending: SendMessageWithMedia.");
                yield break;
            }

            byte[] imageBytes = null;
            var resolvedImagePath = mediaImagePath ?? string.Empty;
            var resolvedAudioPath = mediaAudioPath ?? string.Empty;
            try
            {
                if (string.IsNullOrWhiteSpace(androidLlmModelPath))
                {
                    throw new ArgumentException(
                        $"androidLlmModelPath is required (absolute device path to a media-capable .litertlm). Provide it via {ConfigFileName}.");
                }

                if (!File.Exists(androidLlmModelPath))
                {
                    throw new FileNotFoundException($"LLM model file not found: {androidLlmModelPath}", androidLlmModelPath);
                }

                if (!string.IsNullOrWhiteSpace(resolvedImagePath) && !File.Exists(resolvedImagePath))
                {
                    throw new FileNotFoundException($"Stage-1 image file not found: {resolvedImagePath}", resolvedImagePath);
                }

                if (!string.IsNullOrWhiteSpace(resolvedAudioPath) && !File.Exists(resolvedAudioPath))
                {
                    throw new FileNotFoundException($"Stage-1 audio file not found: {resolvedAudioPath}", resolvedAudioPath);
                }

                if (string.IsNullOrWhiteSpace(resolvedImagePath) && testImage != null)
                {
                    imageBytes = testImage.EncodeToPNG();
                }

                if (string.IsNullOrWhiteSpace(resolvedImagePath) && imageBytes == null && string.IsNullOrWhiteSpace(resolvedAudioPath))
                {
                    throw new ArgumentException(
                        "No stage-1 media configured: set mediaImagePath/mediaAudioPath (absolute device paths) or assign testImage.");
                }
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
                yield break;
            }

            WriteStatus("LLM_MODEL_READY", DescribeFile(androidLlmModelPath));
            yield return null;

            using (var client = new LiteRtLmUnityClient())
            {
            try
            {
                WriteStatus(
                    "INITIALIZE",
                    $"model={androidLlmModelPath}, backend={androidBackend}, visionBackend={androidVisionBackend}, audioBackend={androidAudioBackend}, " +
                    $"maxNumTokens={androidMaxNumTokens}, maxNumImages={androidMaxNumImages}, cacheDir={Application.temporaryCachePath}");
                var initializeStartedAt = Time.realtimeSinceStartup;
                client.Initialize(
                    androidLlmModelPath,
                    androidBackend,
                    Application.temporaryCachePath,
                    androidMaxNumTokens,
                    androidMaxNumImages,
                    systemInstruction: BuildSystemMessage(),
                    visionBackend: androidVisionBackend,
                    audioBackend: androidAudioBackend);
                WriteStatus(
                    "INITIALIZED",
                    $"isInitialized={client.IsInitialized}, elapsedSeconds={Time.realtimeSinceStartup - initializeStartedAt:0.###}");
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
                yield break;
            }

            yield return null;

            try
            {
                var mediaDescription =
                    $"imagePath={(string.IsNullOrWhiteSpace(resolvedImagePath) ? "(none)" : resolvedImagePath)}, " +
                    $"imageBytes={(imageBytes == null ? 0 : imageBytes.Length)}, " +
                    $"audioPath={(string.IsNullOrWhiteSpace(resolvedAudioPath) ? "(none)" : resolvedAudioPath)}";
                WriteStatus("STAGE1_MEDIA_INPUT", mediaDescription);

                var prompt = BuildAndroidFunctionCallingPrompt();
                WriteStatus("LLM_PROMPT", OneLine(Truncate(prompt, 1600)));

                var llmStartedAt = Time.realtimeSinceStartup;
                var rawFunctionCall = client.SendMessageWithMedia(prompt, imageBytes, resolvedImagePath, resolvedAudioPath);
                var llmElapsedSeconds = Time.realtimeSinceStartup - llmStartedAt;
                if (string.IsNullOrWhiteSpace(rawFunctionCall))
                {
                    throw new InvalidOperationException("LLM function-calling response was empty.");
                }

                var toolName = ParseToolName(rawFunctionCall);
                WriteStatus(
                    "FUNCTION_CALL",
                    $"elapsedSeconds={llmElapsedSeconds:0.###}, tool={toolName}, raw={OneLine(Truncate(rawFunctionCall, 1200))}");

                if (!string.Equals(expectedToolName, toolName, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        $"Unexpected function call. expected={expectedToolName}, actual={toolName}, raw={OneLine(Truncate(rawFunctionCall, 1200))}");
                }

                var totalElapsedSeconds = Time.realtimeSinceStartup - demoStartedAt;
                WriteStatus(
                    "SUCCESS",
                    $"tool={toolName}, utterance={textFallbackUtterance}, totalElapsedSeconds={totalElapsedSeconds:0.###}");
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
            }
            }
        }

        // The Android bridge has no separate tools/constrained-decoding channel, so the tools
        // JSON rides inside the prompt (same contract the Windows degraded mode feeds the CLI).
        private string BuildAndroidFunctionCallingPrompt()
        {
            return "Available tools (JSON):\n" + BuildToolsJson() + "\n\n" +
                   "Respond with exactly one JSON object of the form {\"tool\":\"<name>\",\"parameters\":{...}} and nothing else.\n" +
                   "현재 시간: " + ReferenceNow + "\n" +
                   "사용자 발화: " + textFallbackUtterance;
        }

        private void LoadAndroidRuntimeConfig()
        {
            var configPath = Path.Combine(Application.persistentDataPath, ConfigFileName);
            if (!File.Exists(configPath))
            {
                WriteStatus("CONFIG", $"no runtime config at {configPath}; using serialized defaults.");
                return;
            }

            try
            {
                var json = File.ReadAllText(configPath);
                if (TryGetJsonString(json, "llmModelPath", out var configuredModelPath))
                {
                    androidLlmModelPath = configuredModelPath;
                }
                if (TryGetJsonString(json, "backend", out var configuredBackend))
                {
                    androidBackend = configuredBackend;
                }
                if (TryGetJsonString(json, "visionBackend", out var configuredVisionBackend))
                {
                    androidVisionBackend = configuredVisionBackend;
                }
                if (TryGetJsonString(json, "audioBackend", out var configuredAudioBackend))
                {
                    androidAudioBackend = configuredAudioBackend;
                }
                if (TryGetJsonInt(json, "maxNumTokens", out var configuredMaxNumTokens) && configuredMaxNumTokens > 0)
                {
                    androidMaxNumTokens = configuredMaxNumTokens;
                }
                if (TryGetJsonInt(json, "maxNumImages", out var configuredMaxNumImages) && configuredMaxNumImages >= 0)
                {
                    androidMaxNumImages = configuredMaxNumImages;
                }
                if (TryGetJsonString(json, "mediaImagePath", out var configuredImagePath))
                {
                    mediaImagePath = configuredImagePath;
                }
                if (TryGetJsonString(json, "mediaAudioPath", out var configuredAudioPath))
                {
                    mediaAudioPath = configuredAudioPath;
                }
                if (TryGetJsonString(json, "utterance", out var configuredUtterance))
                {
                    textFallbackUtterance = configuredUtterance;
                }
                if (TryGetJsonString(json, "expectedToolName", out var configuredExpectedToolName))
                {
                    expectedToolName = configuredExpectedToolName;
                }

                WriteStatus(
                    "CONFIG",
                    $"loaded {configPath}: model={androidLlmModelPath}, backend={androidBackend}, visionBackend={androidVisionBackend}, " +
                    $"audioBackend={androidAudioBackend}, maxNumTokens={androidMaxNumTokens}, maxNumImages={androidMaxNumImages}, " +
                    $"image={mediaImagePath}, audio={mediaAudioPath}, utterance={textFallbackUtterance}, expectedTool={expectedToolName}");
            }
            catch (Exception ex)
            {
                WriteStatus("CONFIG_ERROR", $"failed to parse {configPath}: {ex.Message}; using serialized defaults.");
            }
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
#endif

#if !UNITY_ANDROID || UNITY_EDITOR
        private IEnumerator RunWindowsTextDegradedPipeline()
        {
            var demoStartedAt = Time.realtimeSinceStartup;
            string executablePath;
            string resolvedModelPath;
            try
            {
                executablePath = ResolveProjectPath(windowsCliExecutablePath);
                if (!File.Exists(executablePath))
                {
                    throw new FileNotFoundException($"Windows CLI executable not found: {executablePath}", executablePath);
                }

                resolvedModelPath = ResolveModelPath(llmModelPath);
                if (!File.Exists(resolvedModelPath))
                {
                    throw new FileNotFoundException($"LLM model file not found: {resolvedModelPath}", resolvedModelPath);
                }
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
                yield break;
            }

            WriteStatus("LLM_MODEL_READY", DescribeFile(resolvedModelPath));
            WriteStatus(
                "TEXT_DEGRADED_MODE",
                "Stage 1 multimodal input is pending the bridge media API; using the configured text fallback utterance instead.");
            WriteStatus("STAGE1_TEXT_INPUT", textFallbackUtterance);

            var prompt = "현재 시간: " + ReferenceNow + "\n사용자 발화: " + textFallbackUtterance;
            WriteStatus("LLM_PROMPT", OneLine(prompt));

            var client = new LiteRtLmWindowsCliClient();
            var llmStartedAt = Time.realtimeSinceStartup;
            var task = client.SendMessageAsync(
                executablePath,
                resolvedModelPath,
                prompt,
                windowsBackend,
                TimeSpan.FromSeconds(Mathf.Max(1f, timeoutSeconds)),
                CancellationToken.None,
                systemMessage: BuildSystemMessage(),
                toolsJson: BuildToolsJson(),
                enableConstrainedDecoding: enableConstrainedDecoding,
                outputMessageJson: outputMessageJson);

            while (!task.IsCompleted)
            {
                yield return null;
            }

            if (task.IsFaulted || task.IsCanceled)
            {
                WriteFailure(task.Exception?.GetBaseException() ?? new InvalidOperationException("Windows CLI task failed."));
                yield break;
            }

            try
            {
                var llmElapsedSeconds = Time.realtimeSinceStartup - llmStartedAt;
                var rawFunctionCall = task.Result;
                if (string.IsNullOrWhiteSpace(rawFunctionCall))
                {
                    throw new InvalidOperationException("LLM function-calling response was empty.");
                }

                var toolName = ParseToolName(rawFunctionCall);
                WriteStatus(
                    "FUNCTION_CALL",
                    $"elapsedSeconds={llmElapsedSeconds:0.###}, tool={toolName}, raw={OneLine(Truncate(rawFunctionCall, 1200))}");

                if (!string.Equals(expectedToolName, toolName, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        $"Unexpected function call. expected={expectedToolName}, actual={toolName}, raw={OneLine(Truncate(rawFunctionCall, 1200))}");
                }

                var totalElapsedSeconds = Time.realtimeSinceStartup - demoStartedAt;
                WriteStatus(
                    "SUCCESS",
                    $"tool={toolName}, utterance={textFallbackUtterance}, totalElapsedSeconds={totalElapsedSeconds:0.###}");
            }
            catch (Exception ex)
            {
                WriteFailure(ex);
            }
        }
#endif

        private static string BuildSystemMessage()
        {
            return "You are a deterministic function-calling router for a Unity command UI.\n" +
                   "Select exactly one tool from the provided tools.\n" +
                   "Current time is " + ReferenceNow + ".\n" +
                   "For date ranges, output full-day or full-month ranges in YYYY-MM-DD HH:MM:SS.\n" +
                   "Use DefaultResponse for unrelated requests.\n" +
                   "Do not explain your choice.";
        }

        private static string BuildToolsJson()
        {
            return @"[
  {""type"":""function"",""function"":{""name"":""IncreaseBrightness"",""description"":""디스플레이/화면 밝기를 더욱 밝게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""DecreaseBrightness"",""description"":""디스플레이의 화면 밝기를 더욱 어둡게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""DecreaseVolume"",""description"":""시스템의 음량을 더욱 작게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""IncreaseVolume"",""description"":""시스템의 음량을 더욱 크게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""ShowMultimodalDataList"",""description"":""멀티모달 데이터 목록을 화면에 표시합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""VisualizeSchedulingResults"",""description"":""타격 스케쥴링 결과를 화면에 가시화합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""VisualizeRecentSituationAwarenessResults"",""description"":""최근 상황인지 결과를 화면에 가시화합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""SortSituationMap"",""description"":""상황도를 정렬합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""HideMultimodalData"",""description"":""멀티모달 데이터를 화면에서 끕니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""ViewSituationAwarenessResults"",""description"":""특정 시간 범위의 상황인지 결과를 열람합니다."",""parameters"":{""type"":""object"",""properties"":{""startTime"":{""type"":""string"",""description"":""시작 시간, YYYY-MM-DD HH:MM:SS""},""endTime"":{""type"":""string"",""description"":""종료 시간, YYYY-MM-DD HH:MM:SS""}},""required"":[""startTime"",""endTime""]}}},
  {""type"":""function"",""function"":{""name"":""ViewThreatAssessmentResults"",""description"":""특정 시간 범위의 위협평가 결과를 열람합니다."",""parameters"":{""type"":""object"",""properties"":{""startTime"":{""type"":""string"",""description"":""시작 시간, YYYY-MM-DD HH:MM:SS""},""endTime"":{""type"":""string"",""description"":""종료 시간, YYYY-MM-DD HH:MM:SS""}},""required"":[""startTime"",""endTime""]}}},
  {""type"":""function"",""function"":{""name"":""ViewSchedulingResults"",""description"":""특정 시간 범위의 스케쥴링 결과를 열람합니다."",""parameters"":{""type"":""object"",""properties"":{""startTime"":{""type"":""string"",""description"":""시작 시간, YYYY-MM-DD HH:MM:SS""},""endTime"":{""type"":""string"",""description"":""종료 시간, YYYY-MM-DD HH:MM:SS""}},""required"":[""startTime"",""endTime""]}}},
  {""type"":""function"",""function"":{""name"":""DefaultResponse"",""description"":""input과 일치하는 description이 없다면 이 함수를 사용합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}}
]";
        }

        private static string ParseToolName(string raw)
        {
            var text = StripBenchmarkInfo(raw ?? string.Empty);
            var tool = MatchFirst(text,
                @"""tool""\s*:\s*""(?<value>[^""]+)""",
                @"""name""\s*:\s*""(?<value>[^""]+)""");
            if (!string.IsNullOrWhiteSpace(tool))
            {
                return tool;
            }

            var contentText = ConcatenateTextFragments(text);
            return MatchFirst(contentText,
                @"""tool""\s*:\s*""(?<value>[^""]+)""",
                @"""name""\s*:\s*""(?<value>[^""]+)""");
        }

        private static string ConcatenateTextFragments(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            foreach (Match match in Regex.Matches(text, @"""text""\s*:\s*""(?<value>[^""]*)"""))
            {
                builder.Append(match.Groups["value"].Value);
            }

            return builder.ToString();
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

        private static string ResolveProjectPath(string configuredPath)
        {
            if (Path.IsPathRooted(configuredPath))
            {
                return configuredPath;
            }

            var candidateInProject = Path.Combine(GetProjectRoot(), configuredPath);
            if (File.Exists(candidateInProject))
            {
                return candidateInProject;
            }

            var repoRoot = Directory.GetParent(GetProjectRoot())?.FullName;
            if (!string.IsNullOrWhiteSpace(repoRoot))
            {
                var candidateInRepo = Path.Combine(repoRoot, configuredPath);
                if (File.Exists(candidateInRepo))
                {
                    return candidateInRepo;
                }
            }

            return candidateInProject;
        }

        private static string ResolveModelPath(string configuredPath)
        {
            if (Path.IsPathRooted(configuredPath))
            {
                return configuredPath;
            }

            var streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, configuredPath);
            if (File.Exists(streamingAssetsPath))
            {
                return streamingAssetsPath;
            }

            return Path.Combine(GetProjectRoot(), "Assets", "StreamingAssets", configuredPath);
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
                var statusPath = GetStatusPath();
                var statusDirectory = Path.GetDirectoryName(statusPath);
                if (!string.IsNullOrWhiteSpace(statusDirectory))
                {
                    Directory.CreateDirectory(statusDirectory);
                }

                File.AppendAllText(statusPath, $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"{LogPrefix} failed to write status file: {ex.Message}");
            }
        }

        private static string GetStatusPath()
        {
            if (Application.isEditor)
            {
                return Path.Combine(GetProjectRoot(), "Builds", "Logs", StatusFileName);
            }

            return Path.Combine(Application.persistentDataPath, StatusFileName);
        }

        private static string GetProjectRoot()
        {
            var projectRoot = Directory.GetParent(Application.dataPath);
            if (projectRoot == null)
            {
                throw new InvalidOperationException("Failed to resolve Unity project root.");
            }

            return projectRoot.FullName;
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
