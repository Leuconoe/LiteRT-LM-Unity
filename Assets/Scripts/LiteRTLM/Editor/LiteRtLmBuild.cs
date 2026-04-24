using System;
using System.IO;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace LiteRTLM.Unity.Editor
{
    public static class LiteRtLmBuild
    {
        private const string ScenePath = "Assets/Scenes/LiteRtLmSampleScene.unity";
        private const string ConversationTestScenePath = "Assets/Scenes/LiteRtLmConversationTestScene.unity";
        private const string StreamingAssetsModelPath = "Assets/StreamingAssets/model.litertlm";
        private const string WindowsSelfTestModelFileName = "gemma-4-E2B-it.litertlm";
        private const string WindowsSelfTestExecutableRelativePath = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
        private const string WindowsSelfTestPrompt = "Say hello from LiteRT-LM Unity editor self-test.";
        private const string WindowsSelfTestStatusRelativePath = "Builds/Logs/LiteRtLmEditorSelfTest.status.txt";
        private const string ConversationTestStatusRelativePath = "Builds/Logs/LiteRtLmConversationTest.status.txt";
        private const string ConversationRequiredToken = "LRT-CTX-042";

        private static readonly string[] ConversationTestPrompts =
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

        [MenuItem("LiteRT-LM/Build Android APK For AVD")]
        public static void BuildAndroidApkForAvd()
        {
            EnsureTestModelInStreamingAssets();

            var projectRoot = GetProjectRoot();
            var outputDirectory = Path.Combine(projectRoot, "Builds", "Android");
            Directory.CreateDirectory(outputDirectory);
            var outputPath = Path.Combine(outputDirectory, "LiteRtLmSample.apk");

            var previousArchitectures = PlayerSettings.Android.targetArchitectures;
            var previousScriptingBackend = PlayerSettings.GetScriptingBackend(BuildTargetGroup.Android);

            try
            {
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);

                var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                {
                    scenes = new[] { ScenePath },
                    locationPathName = outputPath,
                    target = BuildTarget.Android,
                    options = BuildOptions.None,
                });

                if (report.summary.result != BuildResult.Succeeded)
                {
                    throw new InvalidOperationException(
                        $"Android build failed with result {report.summary.result}. See Unity Editor log for details.");
                }

                Debug.Log($"LiteRT-LM Android APK built successfully: {outputPath}");
            }
            finally
            {
                PlayerSettings.Android.targetArchitectures = previousArchitectures;
                PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, previousScriptingBackend);
            }
        }

        [MenuItem("LiteRT-LM/Run Windows Editor Self-Test")]
        public static void RunWindowsEditorSelfTest()
        {
            ResetWindowsEditorSelfTestStatus();
            var response = ExecuteWindowsEditorSelfTest();
            WriteWindowsEditorSelfTestStatus("SUCCESS", response);
            Debug.Log($"LiteRT-LM Windows editor self-test succeeded. Response preview: {Truncate(response, 240)}");
        }

        public static void RunWindowsEditorSelfTestBatchmode()
        {
            try
            {
                ResetWindowsEditorSelfTestStatus();
                var response = ExecuteWindowsEditorSelfTest();
                WriteWindowsEditorSelfTestStatus("SUCCESS", response);
                Debug.Log($"LiteRT-LM Windows editor self-test succeeded. Response preview: {Truncate(response, 240)}");
                return;
            }
            catch (Exception ex)
            {
                WriteWindowsEditorSelfTestStatus("FAILURE", ex.ToString());
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        [MenuItem("LiteRT-LM/Run Windows Conversation Scene Test")]
        public static void RunWindowsConversationSceneTest()
        {
            ResetConversationTestStatus();
            ExecuteWindowsConversationSceneTest();
        }

        public static void RunWindowsConversationSceneTestBatchmode()
        {
            try
            {
                ResetConversationTestStatus();
                ExecuteWindowsConversationSceneTest();
                Debug.Log("LiteRT-LM Windows conversation scene test succeeded.");
                return;
            }
            catch (Exception ex)
            {
                WriteConversationTestStatus("FAILURE", ex.ToString());
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        private static string ExecuteWindowsEditorSelfTest()
        {
            if (Application.platform != RuntimePlatform.WindowsEditor)
            {
                throw new PlatformNotSupportedException(
                    $"Windows editor self-test requires Unity Windows Editor. Current platform: {Application.platform}");
            }

            var projectRoot = GetProjectRoot();
            var executablePath = Path.Combine(projectRoot, "Tools", "Windows", "litert_lm_main.windows_x86_64.exe");
            var modelPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", WindowsSelfTestModelFileName);

            if (!File.Exists(executablePath))
            {
                throw new FileNotFoundException(
                    $"Windows CLI executable not found: {executablePath}",
                    executablePath);
            }

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException(
                    $"StreamingAssets model not found: {modelPath}",
                    modelPath);
            }

            Debug.Log($"LiteRT-LM Windows editor self-test executable: {WindowsSelfTestExecutableRelativePath}");
            Debug.Log($"LiteRT-LM Windows editor self-test model: {modelPath}");
            WriteWindowsEditorSelfTestStatus("INFO", $"Executable={executablePath}");
            WriteWindowsEditorSelfTestStatus("INFO", $"Model={modelPath}");

            var windowsCliClient = new LiteRtLmWindowsCliClient();
            WriteWindowsEditorSelfTestStatus("INFO", "Invoking Windows CLI client.");
            var response = windowsCliClient.SendMessage(
                executablePath,
                modelPath,
                WindowsSelfTestPrompt,
                "cpu");
            WriteWindowsEditorSelfTestStatus("INFO", $"ResponseLength={response?.Length ?? 0}");

            if (string.IsNullOrWhiteSpace(response))
            {
                throw new InvalidOperationException("Windows editor self-test returned an empty response.");
            }

            return response;
        }

        private static void ExecuteWindowsConversationSceneTest()
        {
            if (Application.platform != RuntimePlatform.WindowsEditor)
            {
                throw new PlatformNotSupportedException(
                    $"Windows conversation scene test requires Unity Windows Editor. Current platform: {Application.platform}");
            }

            WriteConversationTestStatus("DOMAIN", "Conversation scene test is running after Unity domain load.");
            var scene = EditorSceneManager.OpenScene(ConversationTestScenePath, OpenSceneMode.Single);
            WriteConversationTestStatus("SCENE", $"Loaded={scene.path}");

            var runner = UnityEngine.Object.FindObjectOfType<LiteRtLmConversationTestRunner>();
            if (runner == null)
            {
                throw new InvalidOperationException($"Conversation test scene does not contain {nameof(LiteRtLmConversationTestRunner)}.");
            }

            var projectRoot = GetProjectRoot();
            var executablePath = Path.Combine(projectRoot, "Tools", "Windows", "litert_lm_main.windows_x86_64.exe");
            var modelPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", WindowsSelfTestModelFileName);
            if (!File.Exists(executablePath))
            {
                throw new FileNotFoundException($"Windows CLI executable not found: {executablePath}", executablePath);
            }

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"StreamingAssets model not found: {modelPath}", modelPath);
            }

            WriteConversationTestStatus("INFO", $"Executable={executablePath}");
            WriteConversationTestStatus("INFO", $"Model={modelPath}");
            WriteConversationTestStatus("INFO", $"Turns={ConversationTestPrompts.Length}");

            var client = new LiteRtLmWindowsCliClient();
            var history = new StringBuilder();
            string lastResponse = null;
            for (var i = 0; i < ConversationTestPrompts.Length; i++)
            {
                var turn = i + 1;
                var prompt = ConversationTestPrompts[i];
                var effectivePrompt = history.Length == 0
                    ? prompt
                    : "아래 [대화 메모]는 내부 참고용입니다. 메모 내용을 복사하지 말고 [현재 요청]에 대한 답변만 작성하세요.\n\n[대화 메모]\n" + history + "\n[현재 요청]\n" + prompt;

                WriteConversationTestStatus("TURN", $"{turn}/{ConversationTestPrompts.Length}: promptLength={prompt.Length}, effectiveLength={effectivePrompt.Length}");
                var startTime = DateTime.UtcNow;
                var response = client.SendMessageAsync(
                        executablePath,
                        modelPath,
                        effectivePrompt,
                        "cpu",
                        TimeSpan.FromSeconds(120),
                        CancellationToken.None)
                    .GetAwaiter()
                    .GetResult();
                var elapsedSeconds = (DateTime.UtcNow - startTime).TotalSeconds;

                if (string.IsNullOrWhiteSpace(response))
                {
                    throw new InvalidOperationException($"Conversation scene test returned an empty response on turn {turn}.");
                }

                lastResponse = response;
                WriteConversationTestStatus("RESPONSE", $"{turn}/{ConversationTestPrompts.Length}: elapsedSeconds={elapsedSeconds:0.#}, length={response.Length}, preview={Truncate(response, 180)}");
                history.Append("Turn ");
                history.Append(turn);
                history.Append(" user: ");
                history.AppendLine(prompt);
                history.Append("Turn ");
                history.Append(turn);
                history.Append(" assistant summary: ");
                history.AppendLine(Truncate(OneLine(StripBenchmarkInfo(response)), 180));
            }

            if (!ValidateConversationFinalResponse(lastResponse))
            {
                throw new InvalidOperationException($"Final response did not cleanly include required token: {ConversationRequiredToken}");
            }

            WriteConversationTestStatus("SUCCESS", $"Completed {ConversationTestPrompts.Length} turns.");
        }

        private static void EnsureTestModelInStreamingAssets()
        {
            var projectRoot = GetProjectRoot();
            var repoRootDirectory = Directory.GetParent(projectRoot);
            if (repoRootDirectory == null)
            {
                throw new InvalidOperationException("Failed to resolve repository root from Unity project path.");
            }

            var sourceModelPath = Path.Combine(repoRootDirectory.FullName, "runtime", "testdata", "test_lm.litertlm");
            if (!File.Exists(sourceModelPath))
            {
                throw new FileNotFoundException($"Test model not found: {sourceModelPath}", sourceModelPath);
            }

            var destinationPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", "model.litertlm");
            var destinationDirectory = Path.GetDirectoryName(destinationPath);
            if (string.IsNullOrWhiteSpace(destinationDirectory))
            {
                throw new InvalidOperationException("Failed to resolve StreamingAssets directory path.");
            }

            Directory.CreateDirectory(destinationDirectory);
            File.Copy(sourceModelPath, destinationPath, true);
            AssetDatabase.ImportAsset(StreamingAssetsModelPath, ImportAssetOptions.ForceUpdate);
            AssetDatabase.Refresh();
        }

        private static string Truncate(string value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
            {
                return value;
            }

            return value.Substring(0, maxLength) + "...";
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

        private static bool ValidateConversationFinalResponse(string response)
        {
            var cleanedResponse = StripBenchmarkInfo(response);
            return cleanedResponse.Contains(ConversationRequiredToken) &&
                   cleanedResponse.Length <= 120 &&
                   !cleanedResponse.Contains("User:") &&
                   !cleanedResponse.Contains("Assistant:") &&
                   !cleanedResponse.Contains("[대화 메모]");
        }

        private static string OneLine(string value)
        {
            return string.IsNullOrEmpty(value)
                ? string.Empty
                : value.Replace("\r", " ").Replace("\n", " ").Trim();
        }

        private static void ResetWindowsEditorSelfTestStatus()
        {
            var statusPath = GetWindowsEditorSelfTestStatusPath();
            var statusDirectory = Path.GetDirectoryName(statusPath);
            if (!string.IsNullOrWhiteSpace(statusDirectory))
            {
                Directory.CreateDirectory(statusDirectory);
            }

            File.WriteAllText(statusPath, string.Empty);
            WriteWindowsEditorSelfTestStatus("START", DateTime.UtcNow.ToString("O"));
        }

        private static void WriteWindowsEditorSelfTestStatus(string phase, string message)
        {
            var statusPath = GetWindowsEditorSelfTestStatusPath();
            var line = $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}";
            File.AppendAllText(statusPath, line);
        }

        private static string GetWindowsEditorSelfTestStatusPath()
        {
            return Path.Combine(GetProjectRoot(), "Builds", "Logs", "LiteRtLmEditorSelfTest.status.txt");
        }

        private static void ResetConversationTestStatus()
        {
            var statusPath = GetConversationTestStatusPath();
            var statusDirectory = Path.GetDirectoryName(statusPath);
            if (!string.IsNullOrWhiteSpace(statusDirectory))
            {
                Directory.CreateDirectory(statusDirectory);
            }

            File.WriteAllText(statusPath, string.Empty);
            WriteConversationTestStatus("START", DateTime.UtcNow.ToString("O"));
        }

        private static void WriteConversationTestStatus(string phase, string message)
        {
            var statusPath = GetConversationTestStatusPath();
            var line = $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}";
            Debug.Log($"[LiteRT-LM ConversationTest] {phase}: {message}");
            File.AppendAllText(statusPath, line);
        }

        private static string GetConversationTestStatusPath()
        {
            return Path.Combine(GetProjectRoot(), ConversationTestStatusRelativePath);
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
