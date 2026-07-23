using System;
using System.Collections;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmConversationTestRunner : MonoBehaviour
    {
        [SerializeField] private bool runOnStart = true;
        [SerializeField] private string modelPath = "Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm";
        [SerializeField] private string windowsCliExecutablePath = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
        [SerializeField] private string windowsBackend = "GPU";
        [SerializeField] private float timeoutSeconds = 120f;
        [SerializeField] private bool includeConversationHistory = true;
        [SerializeField] private string requiredFinalResponseToken = "LRT-CTX-042";
        [SerializeField]
        private string[] testPrompts =
        {
            "짧게 인사해 주세요.",
            "Unity Editor 안에서 실행 중인 LiteRT-LM 테스트입니다. 한 문장으로 현재 상태를 설명해 주세요.",
            "다음 숫자들을 더한 값만 답하세요: 17, 23, 60.",
            "내가 기억시킬 코드워드는 LRT-CTX-042 입니다. 코드워드를 기억했다는 한 문장만 답하세요.",
            "한글 입력 검증입니다. '한글 프롬프트를 이해했습니다'라는 문장을 그대로 포함해서 답하세요.",
            "영어와 한국어를 섞어 한 문장으로 답하세요: Unity sample is running.",
            "아래 요구사항을 3개 bullet로 요약하세요: 모델 경로 확인, Windows CLI 실행, 두 번째 응답 확인.",
            "조금 긴 요청입니다. 사용자가 Unity 샘플 씬에서 첫 번째 대화 후 두 번째 대화 응답이 보이지 않는다고 보고했습니다. 가능한 원인을 UI 상태, CLI 실행, 응답 표시의 세 관점으로 짧게 나누어 설명해 주세요.",
            "이전 대화에서 내가 기억시키라고 한 코드워드가 무엇인지 답하세요. 답변에는 코드워드를 반드시 포함하세요.",
            "마지막 검증입니다. 대화 메모에 있는 코드워드만 정확히 한 번 출력하세요. 다른 설명은 하지 마세요.",
        };

        public string Status { get; private set; } = "Idle";
        public string[] Responses { get; private set; } = Array.Empty<string>();

        private void Start()
        {
            if (runOnStart)
            {
                StartCoroutine(RunConversationTestRoutine());
            }
        }

        public IEnumerator RunConversationTestRoutine()
        {
            Status = "Running";
            Responses = new string[testPrompts?.Length ?? 0];
            ResetStatus();
            LogStatus("START", $"{DateTime.UtcNow:O}, turns={Responses.Length}, timeoutSeconds={timeoutSeconds:0.#}, includeHistory={includeConversationHistory}");

            string executablePath;
            string resolvedModelPath;
            try
            {
                executablePath = ResolveProjectPath(windowsCliExecutablePath);
                resolvedModelPath = ResolveModelPath(modelPath);
                if (!File.Exists(executablePath))
                {
                    throw new FileNotFoundException($"Windows CLI executable not found: {executablePath}", executablePath);
                }

                if (!File.Exists(resolvedModelPath))
                {
                    throw new FileNotFoundException($"Model file not found: {resolvedModelPath}", resolvedModelPath);
                }

                LogStatus("INFO", $"Executable={executablePath}");
                LogStatus("INFO", $"Model={resolvedModelPath}");
            }
            catch (Exception ex)
            {
                Status = $"Failed: {ex.Message}";
                LogStatus("FAILURE", ex.ToString());
                Debug.LogException(ex);
                yield break;
            }

            var client = new LiteRtLmWindowsCliClient();
            var conversationHistory = new StringBuilder();
            for (var i = 0; i < Responses.Length; i++)
            {
                var prompt = testPrompts[i] ?? string.Empty;
                var effectivePrompt = BuildPromptWithHistory(conversationHistory, prompt);
                var turnNumber = i + 1;
                var turnStartTime = DateTime.UtcNow;
                var nextProgressLogTime = turnStartTime.AddSeconds(5);
                LogStatus("TURN", $"{turnNumber}/{Responses.Length}: promptLength={prompt.Length}, effectiveLength={effectivePrompt.Length}");
                var task = client.SendMessageAsync(
                    executablePath,
                    resolvedModelPath,
                    effectivePrompt,
                    windowsBackend,
                    TimeSpan.FromSeconds(Mathf.Max(1f, timeoutSeconds)),
                    CancellationToken.None);

                while (!task.IsCompleted)
                {
                    if (DateTime.UtcNow >= nextProgressLogTime)
                    {
                        var elapsedSeconds = (DateTime.UtcNow - turnStartTime).TotalSeconds;
                        LogStatus("WAIT", $"{turnNumber}/{Responses.Length}: elapsedSeconds={elapsedSeconds:0.#}, timeoutSeconds={timeoutSeconds:0.#}");
                        nextProgressLogTime = DateTime.UtcNow.AddSeconds(5);
                    }

                    yield return null;
                }

                var turnElapsedSeconds = (DateTime.UtcNow - turnStartTime).TotalSeconds;

                if (task.IsFaulted)
                {
                    var ex = task.Exception?.GetBaseException() ?? new InvalidOperationException("Conversation test task failed.");
                    Status = $"Failed on turn {i + 1}: {ex.Message}";
                    LogStatus("FAILURE", $"{turnNumber}/{Responses.Length}: elapsedSeconds={turnElapsedSeconds:0.#}, error={ex}");
                    Debug.LogException(ex);
                    yield break;
                }

                if (task.IsCanceled)
                {
                    Status = $"Canceled on turn {i + 1}";
                    LogStatus("FAILURE", $"{turnNumber}/{Responses.Length}: elapsedSeconds={turnElapsedSeconds:0.#}, {Status}");
                    yield break;
                }

                Responses[i] = task.Result;
                LogStatus("RESPONSE", $"{turnNumber}/{Responses.Length}: elapsedSeconds={turnElapsedSeconds:0.#}, length={Responses[i]?.Length ?? 0}, preview={Truncate(Responses[i], 180)}");
                if (string.IsNullOrWhiteSpace(Responses[i]))
                {
                    Status = $"Failed on turn {i + 1}: empty response";
                    LogStatus("FAILURE", Status);
                    yield break;
                }

                AppendConversationTurn(conversationHistory, turnNumber, prompt, Responses[i]);
            }

            if (!ValidateFinalResponse(Responses.Length == 0 ? string.Empty : Responses[Responses.Length - 1]))
            {
                Status = $"Failed: final response did not cleanly include {requiredFinalResponseToken}";
                LogStatus("FAILURE", Status);
                yield break;
            }

            Status = "Completed";
            LogStatus("SUCCESS", $"Completed {Responses.Length} turns.");
            Debug.Log($"LiteRT-LM conversation test completed. Turns={Responses.Length}");
        }

        private string BuildPromptWithHistory(StringBuilder conversationHistory, string prompt)
        {
            if (!includeConversationHistory || conversationHistory.Length == 0)
            {
                return prompt;
            }

            return "아래 [대화 메모]는 내부 참고용입니다. 메모 내용을 복사하지 말고 [현재 요청]에 대한 답변만 작성하세요.\n\n[대화 메모]\n" +
                   conversationHistory +
                   "\n[현재 요청]\n" + prompt;
        }

        private static void AppendConversationTurn(StringBuilder conversationHistory, int turnNumber, string prompt, string response)
        {
            conversationHistory.Append("Turn ");
            conversationHistory.Append(turnNumber);
            conversationHistory.Append(" user: ");
            conversationHistory.AppendLine(prompt);
            conversationHistory.Append("Turn ");
            conversationHistory.Append(turnNumber);
            conversationHistory.Append(" assistant summary: ");
            conversationHistory.AppendLine(Truncate(OneLine(StripBenchmarkInfo(response)), 180));
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

        private bool ValidateFinalResponse(string response)
        {
            if (string.IsNullOrWhiteSpace(requiredFinalResponseToken))
            {
                return true;
            }

            var cleanedResponse = StripBenchmarkInfo(response);
            if (!cleanedResponse.Contains(requiredFinalResponseToken))
            {
                return false;
            }

            return cleanedResponse.Length <= 120 &&
                   !cleanedResponse.Contains("User:") &&
                   !cleanedResponse.Contains("Assistant:") &&
                   !cleanedResponse.Contains("[대화 메모]");
        }

        private static string ResolveProjectPath(string configuredPath)
        {
            if (Path.IsPathRooted(configuredPath))
            {
                return configuredPath;
            }

            return Path.Combine(GetProjectRoot(), configuredPath);
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

        private static void WriteStatus(string phase, string message)
        {
            var statusPath = Path.Combine(GetProjectRoot(), "Builds", "Logs", "LiteRtLmConversationTest.status.txt");
            var directory = Path.GetDirectoryName(statusPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.AppendAllText(statusPath, $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}");
        }

        private static void LogStatus(string phase, string message)
        {
            Debug.Log($"[LiteRT-LM ConversationTest] {phase}: {message}");
            WriteStatus(phase, message);
        }

        private static void ResetStatus()
        {
            var statusPath = Path.Combine(GetProjectRoot(), "Builds", "Logs", "LiteRtLmConversationTest.status.txt");
            var directory = Path.GetDirectoryName(statusPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(statusPath, string.Empty);
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

        private static string GetProjectRoot()
        {
            var projectRoot = Directory.GetParent(Application.dataPath);
            if (projectRoot == null)
            {
                throw new InvalidOperationException("Failed to resolve Unity project root.");
            }

            return projectRoot.FullName;
        }
    }
}
