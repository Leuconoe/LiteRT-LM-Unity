using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace LiteRTLM.Unity.Editor
{
    public static class LiteRtLmBuild
    {
        private const string ScenePath = "Assets/Scenes/LiteRtLmSampleScene.unity";
        private const string AndroidSmokeTestScenePath = "Assets/Scenes/LiteRtLmAndroidSmokeTestScene.unity";
        private const string AndroidSmokeBuildScenePath = "Assets/Scenes/LiteRtLmAndroidSmokeTestBuildScene.generated.unity";
        private const string ConversationTestScenePath = "Assets/Scenes/LiteRtLmConversationTestScene.unity";
        private const string FunctionCallingBenchmarkScenePath = "Assets/Scenes/LiteRtLmFunctionCallingBenchmarkScene.unity";
        private const string StreamingAssetsModelPath = "Assets/StreamingAssets/model.litertlm";
        private const string WindowsSelfTestModelFileName = "gemma-4-E2B-it.litertlm";
        private const string WindowsSelfTestExecutableRelativePath = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
        private const string WindowsSelfTestPrompt = "Say hello from LiteRT-LM Unity editor self-test.";
        private const string WindowsSelfTestStatusRelativePath = "Builds/Logs/LiteRtLmEditorSelfTest.status.txt";
        private const string ConversationTestStatusRelativePath = "Builds/Logs/LiteRtLmConversationTest.status.txt";
        private const string FunctionCallingBenchmarkStatusRelativePath = "Builds/Logs/LiteRtLmFunctionCallingBenchmark.status.txt";
        private const string ConversationRequiredToken = "LRT-CTX-042";
        private const double FunctionCallingBenchmarkConsoleMirrorIntervalSeconds = 1.0;
        private const string FunctionCallingGemma4ModelPath = "gemma-4-E2B-it.litertlm";
        private const string FunctionCallingGemma1BModelPath = "gemma3-1b-it-int4.litertlm";
        private const string FunctionCallingGemma270MModelPath = "gemma3-270m-it-q8.litertlm";
        private const string FunctionCallingQwen25ModelPath = "Qwen2.5-0.5B-Instruct-q8.litertlm";
        private const string FunctionCallingQwen25_1_5BModelPath = "Qwen2.5-1.5B-Instruct-q8.litertlm";
        private const string FunctionCallingQwen3ModelPath = "Qwen3-0.6B.litertlm";
        private const string FunctionCallingMobileActionsModelPath = "mobile_actions_q8_ekv1024.litertlm";

        private static bool functionCallingBenchmarkConsoleMirrorActive;
        private static long functionCallingBenchmarkConsoleMirrorPosition;
        private static double functionCallingBenchmarkNextConsoleMirrorTime;

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
            var androidBuildTarget = UnityEditor.Build.NamedBuildTarget.Android;
            var previousScriptingBackend = PlayerSettings.GetScriptingBackend(androidBuildTarget);

            try
            {
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, ScriptingImplementation.IL2CPP);

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
                PlayerSettings.SetScriptingBackend(androidBuildTarget, previousScriptingBackend);
            }
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK")]
        public static void BuildAndroidAvdSmokeTestApk()
        {
            EnsureTestModelInStreamingAssets();
            BuildAndroidAvdSmokeTestApk("model.litertlm", "GPU", "LiteRtLmAndroidSmokeTest.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma-4-E2B-it")]
        public static void BuildAndroidAvdSmokeTestApkGemma4()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingGemma4ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it.apk", true, 4000, 0, 128);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke APK/Gemma 4 E2B IT GPU No Speculative")]
        public static void BuildAndroidAvdSmokeTestApkGemma4NoSpeculative()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingGemma4ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-nospec.apk", false, 4000, 0, 128);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke APK/Gemma 4 E2B IT CPU")]
        public static void BuildAndroidAvdSmokeTestApkGemma4Cpu()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingGemma4ModelPath, "CPU", "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-CPU.apk", false, 4000, 0, 128);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma3-1b-it-int4")]
        public static void BuildAndroidAvdSmokeTestApkGemma1B()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingGemma1BModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma3-1b-it-int4 CPU")]
        public static void BuildAndroidAvdSmokeTestApkGemma1BCpu()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingGemma1BModelPath, "CPU", "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-CPU.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma3-270m-it-q8")]
        public static void BuildAndroidAvdSmokeTestApkGemma270M()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingGemma270MModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/mobile_actions_q8_ekv1024")]
        public static void BuildAndroidAvdSmokeTestApkMobileActions()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingMobileActionsModelPath, "GPU", "LiteRtLmAndroidSmokeTest-mobile_actions_q8_ekv1024.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen3-0.6B")]
        public static void BuildAndroidAvdSmokeTestApkQwen3()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingQwen3ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-Qwen3-0.6B.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-0.5B-Instruct")]
        public static void BuildAndroidAvdSmokeTestApkQwen25()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingQwen25ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-0.5B-Instruct CPU")]
        public static void BuildAndroidAvdSmokeTestApkQwen25Cpu()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingQwen25ModelPath, "CPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-1.5B-Instruct")]
        public static void BuildAndroidAvdSmokeTestApkQwen25_1_5B()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingQwen25_1_5BModelPath, "GPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct.apk", false);
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-1.5B-Instruct CPU")]
        public static void BuildAndroidAvdSmokeTestApkQwen25_1_5BCpu()
        {
            BuildAndroidAvdSmokeTestApk(FunctionCallingQwen25_1_5BModelPath, "CPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct-CPU.apk", false);
        }

        private static void BuildAndroidAvdSmokeTestApk(
            string modelFileName,
            string backend,
            string outputFileName,
            bool enableSpeculativeDecoding,
            int maxNumTokens = 64,
            int maxNumImages = 0,
            int benchmarkPrefillTokens = 64)
        {
            if (string.IsNullOrWhiteSpace(modelFileName))
            {
                throw new ArgumentException("Android smoke test model file name is required.", nameof(modelFileName));
            }

            if (string.IsNullOrWhiteSpace(backend))
            {
                throw new ArgumentException("Android smoke test backend is required.", nameof(backend));
            }

            var projectRoot = GetProjectRoot();
            var modelAssetPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", modelFileName);
            if (!File.Exists(modelAssetPath))
            {
                throw new FileNotFoundException($"Android smoke test model not found: {modelAssetPath}", modelAssetPath);
            }

            var outputDirectory = Path.Combine(projectRoot, "Builds", "Android");
            Directory.CreateDirectory(outputDirectory);
            var outputPath = Path.Combine(outputDirectory, outputFileName);
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }

            var previousArchitectures = PlayerSettings.Android.targetArchitectures;
            var androidBuildTarget = UnityEditor.Build.NamedBuildTarget.Android;
            var previousScriptingBackend = PlayerSettings.GetScriptingBackend(androidBuildTarget);
            var previousUseDefaultGraphicsApis = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android);
            var previousGraphicsApis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
            var buildScenePath = CreateAndroidSmokeBuildScene(modelFileName, backend, enableSpeculativeDecoding, maxNumTokens, maxNumImages, benchmarkPrefillTokens);

            try
            {
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, ScriptingImplementation.IL2CPP);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });

                WithAndroidSmokeStreamingAssets(modelFileName, () =>
                {
                    var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                    {
                        scenes = new[] { buildScenePath },
                        locationPathName = outputPath,
                        target = BuildTarget.Android,
                        options = BuildOptions.Development,
                    });

                    if (report.summary.result != BuildResult.Succeeded)
                    {
                        throw new InvalidOperationException(
                            $"Android AVD smoke build failed with result {report.summary.result}. See Unity Editor log for details.");
                    }
                });

                Debug.Log($"LiteRT-LM Android AVD smoke APK built successfully: {outputPath}, model={modelFileName}, backend={backend}, speculative={enableSpeculativeDecoding}, maxNumTokens={maxNumTokens}, maxNumImages={maxNumImages}");
            }
            finally
            {
                DeleteGeneratedAndroidSmokeBuildScene(buildScenePath);
                PlayerSettings.Android.targetArchitectures = previousArchitectures;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, previousScriptingBackend);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, previousUseDefaultGraphicsApis);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, previousGraphicsApis);
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

        [MenuItem("LiteRT-LM/Run Windows Function Calling Benchmark")]
        public static void RunWindowsFunctionCallingBenchmark()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(FunctionCallingGemma4ModelPath);
        }

        [MenuItem("LiteRT-LM/Run Windows Function Calling Benchmark", true)]
        public static bool CanRunWindowsFunctionCallingBenchmark()
        {
            return !EditorApplication.isCompiling &&
                   !EditorApplication.isUpdating &&
                   !EditorApplication.isPlayingOrWillChangePlaymode &&
                   !LiteRtLmFunctionCallingBenchmarkRunner.IsBackgroundRunActive;
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma-4-E2B-it")]
        public static void RunWindowsFunctionCallingBenchmarkGemma4()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(FunctionCallingGemma4ModelPath);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma-4-E2B-it", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkGemma4()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-1b-it-int4")]
        public static void RunWindowsFunctionCallingBenchmarkGemma1B()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(FunctionCallingGemma1BModelPath);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-1b-it-int4", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkGemma1B()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-1b-it-int4 Unconstrained")]
        public static void RunWindowsFunctionCallingBenchmarkGemma1BUnconstrained()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(FunctionCallingGemma1BModelPath, false);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-1b-it-int4 Unconstrained", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkGemma1BUnconstrained()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-270m-it-q8")]
        public static void RunWindowsFunctionCallingBenchmarkGemma270M()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(FunctionCallingGemma270MModelPath);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-270m-it-q8", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkGemma270M()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-270m-it-q8 Unconstrained")]
        public static void RunWindowsFunctionCallingBenchmarkGemma270MUnconstrained()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(FunctionCallingGemma270MModelPath, false);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-270m-it-q8 Unconstrained", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkGemma270MUnconstrained()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-270m-it-q8 Compact Unconstrained")]
        public static void RunWindowsFunctionCallingBenchmarkGemma270MCompactUnconstrained()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(
                FunctionCallingGemma270MModelPath,
                false,
                LiteRtLmFunctionCallingBenchmarkRunner.PromptProfile.Compact);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run gemma3-270m-it-q8 Compact Unconstrained", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkGemma270MCompactUnconstrained()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run Qwen3-0.6B")]
        public static void RunWindowsFunctionCallingBenchmarkQwen3()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(
                FunctionCallingQwen3ModelPath,
                false,
                LiteRtLmFunctionCallingBenchmarkRunner.PromptProfile.QwenHermes);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run Qwen3-0.6B", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkQwen3()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run Qwen3-0.6B Unconstrained")]
        public static void RunWindowsFunctionCallingBenchmarkQwen3Unconstrained()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(
                FunctionCallingQwen3ModelPath,
                false,
                LiteRtLmFunctionCallingBenchmarkRunner.PromptProfile.QwenHermes);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run Qwen3-0.6B Unconstrained", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkQwen3Unconstrained()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run mobile_actions_q8_ekv1024")]
        public static void RunWindowsFunctionCallingBenchmarkMobileActions()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(
                FunctionCallingMobileActionsModelPath,
                true,
                LiteRtLmFunctionCallingBenchmarkRunner.PromptProfile.MobileActions);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run mobile_actions_q8_ekv1024", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkMobileActions()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run mobile_actions_q8_ekv1024 Unconstrained")]
        public static void RunWindowsFunctionCallingBenchmarkMobileActionsUnconstrained()
        {
            StartWindowsFunctionCallingBenchmarkInBackground(
                FunctionCallingMobileActionsModelPath,
                false,
                LiteRtLmFunctionCallingBenchmarkRunner.PromptProfile.MobileActions);
        }

        [MenuItem("LiteRT-LM/Function Calling Benchmark/Run mobile_actions_q8_ekv1024 Unconstrained", true)]
        public static bool CanRunWindowsFunctionCallingBenchmarkMobileActionsUnconstrained()
        {
            return CanRunWindowsFunctionCallingBenchmark();
        }

        public static void RunWindowsFunctionCallingBenchmarkBatchmode()
        {
            try
            {
                ExecuteWindowsFunctionCallingBenchmark();
                Debug.Log("LiteRT-LM Windows function-calling benchmark succeeded.");
                return;
            }
            catch (Exception ex)
            {
                WriteFunctionCallingBenchmarkStatus("FAILURE", ex.ToString());
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

            var runner = UnityEngine.Object.FindAnyObjectByType<LiteRtLmConversationTestRunner>();
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

        private static void ExecuteWindowsFunctionCallingBenchmark()
        {
            if (Application.platform != RuntimePlatform.WindowsEditor)
            {
                throw new PlatformNotSupportedException(
                    $"Windows function-calling benchmark requires Unity Windows Editor. Current platform: {Application.platform}");
            }

            ResetFunctionCallingBenchmarkStatus();
            WriteFunctionCallingBenchmarkStatus("DOMAIN", "Function-calling benchmark is running after Unity domain load.");
            var scene = EditorSceneManager.OpenScene(FunctionCallingBenchmarkScenePath, OpenSceneMode.Single);
            WriteFunctionCallingBenchmarkStatus("SCENE", $"Loaded={scene.path}");

            var runner = UnityEngine.Object.FindAnyObjectByType<LiteRtLmFunctionCallingBenchmarkRunner>();
            if (runner == null)
            {
                throw new InvalidOperationException($"Benchmark scene does not contain {nameof(LiteRtLmFunctionCallingBenchmarkRunner)}.");
            }

            var summary = runner.RunBenchmarkBlocking();
            WriteFunctionCallingBenchmarkStatus("SUCCESS", $"Passed={summary.Passed}, Failed={summary.Failed}, Accuracy={summary.Accuracy:0.000}");
        }

        private static void StartWindowsFunctionCallingBenchmarkInBackground(string modelPath)
        {
            StartWindowsFunctionCallingBenchmarkInBackground(modelPath, true);
        }

        private static void StartWindowsFunctionCallingBenchmarkInBackground(string modelPath, bool enableConstrainedDecoding)
        {
            StartWindowsFunctionCallingBenchmarkInBackground(
                modelPath,
                enableConstrainedDecoding,
                LiteRtLmFunctionCallingBenchmarkRunner.PromptProfile.CurrentTuned);
        }

        private static void StartWindowsFunctionCallingBenchmarkInBackground(
            string modelPath,
            bool enableConstrainedDecoding,
            LiteRtLmFunctionCallingBenchmarkRunner.PromptProfile promptProfile)
        {
            if (Application.platform != RuntimePlatform.WindowsEditor)
            {
                throw new PlatformNotSupportedException(
                    $"Windows function-calling benchmark requires Unity Windows Editor. Current platform: {Application.platform}");
            }

            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Debug.LogWarning("LiteRT-LM function-calling benchmark skipped because Unity is entering or running Play Mode.");
                return;
            }

            WriteFunctionCallingBenchmarkStatus("DOMAIN", "Function-calling benchmark is starting as a background editor task.");
            var scene = EditorSceneManager.OpenScene(FunctionCallingBenchmarkScenePath, OpenSceneMode.Single);
            WriteFunctionCallingBenchmarkStatus("SCENE", $"Loaded={scene.path}");

            var runner = UnityEngine.Object.FindAnyObjectByType<LiteRtLmFunctionCallingBenchmarkRunner>();
            if (runner == null)
            {
                throw new InvalidOperationException($"Benchmark scene does not contain {nameof(LiteRtLmFunctionCallingBenchmarkRunner)}.");
            }

            if (runner.RunBenchmarkInBackground(
                    modelPath,
                    promptProfile,
                    enableConstrainedDecoding,
                    true))
            {
                StartFunctionCallingBenchmarkConsoleMirror();
                Debug.Log($"LiteRT-LM function-calling benchmark started in the background. Model={modelPath}, constrained={enableConstrainedDecoding}, promptProfile={promptProfile}. Watch {GetFunctionCallingBenchmarkStatusPath()} for progress.");
            }
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

        private static void EnsureAndroidSmokeTestScene()
        {
            if (File.Exists(Path.Combine(GetProjectRoot(), AndroidSmokeTestScenePath)))
            {
                return;
            }

            var activeScene = SceneManager.GetActiveScene();
            var hasSavedActiveScene = activeScene.IsValid() && !string.IsNullOrEmpty(activeScene.path);
            var newSceneMode = Application.isBatchMode || !hasSavedActiveScene
                ? NewSceneMode.Single
                : NewSceneMode.Additive;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, newSceneMode);
            var runnerObject = new GameObject("LiteRtLmAndroidSmokeTestRunner");
            SceneManager.MoveGameObjectToScene(runnerObject, scene);
            runnerObject.AddComponent<LiteRtLmAndroidSmokeTestRunner>();

            var sceneDirectory = Path.GetDirectoryName(Path.Combine(GetProjectRoot(), AndroidSmokeTestScenePath));
            if (!string.IsNullOrWhiteSpace(sceneDirectory))
            {
                Directory.CreateDirectory(sceneDirectory);
            }

            if (!EditorSceneManager.SaveScene(scene, AndroidSmokeTestScenePath))
            {
                throw new InvalidOperationException($"Failed to save Android smoke test scene: {AndroidSmokeTestScenePath}");
            }

            EditorSceneManager.CloseScene(scene, true);
            AssetDatabase.ImportAsset(AndroidSmokeTestScenePath, ImportAssetOptions.ForceUpdate);
            AssetDatabase.Refresh();
        }

        private static string CreateAndroidSmokeBuildScene(
            string modelFileName,
            string backend,
            bool enableSpeculativeDecoding,
            int maxNumTokens,
            int maxNumImages,
            int benchmarkPrefillTokens)
        {
            DeleteGeneratedAndroidSmokeBuildScene(AndroidSmokeBuildScenePath);

            var activeScene = SceneManager.GetActiveScene();
            var hasSavedActiveScene = activeScene.IsValid() && !string.IsNullOrEmpty(activeScene.path);
            var newSceneMode = Application.isBatchMode || !hasSavedActiveScene
                ? NewSceneMode.Single
                : NewSceneMode.Additive;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, newSceneMode);
            try
            {
                var runnerObject = new GameObject("LiteRtLmAndroidSmokeTestRunner");
                SceneManager.MoveGameObjectToScene(runnerObject, scene);
                var runner = runnerObject.AddComponent<LiteRtLmAndroidSmokeTestRunner>();

                var serializedRunner = new SerializedObject(runner);
                var modelPathProperty = serializedRunner.FindProperty("modelPath");
                var backendProperty = serializedRunner.FindProperty("backend");
                var maxNumTokensProperty = serializedRunner.FindProperty("maxNumTokens");
                var maxNumImagesProperty = serializedRunner.FindProperty("maxNumImages");
                var enableSpeculativeDecodingProperty = serializedRunner.FindProperty("enableSpeculativeDecoding");
                var runStandaloneBenchmarkProperty = serializedRunner.FindProperty("runStandaloneBenchmark");
                var benchmarkPrefillTokensProperty = serializedRunner.FindProperty("benchmarkPrefillTokens");
                var benchmarkRunsProperty = serializedRunner.FindProperty("benchmarkRuns");

                if (modelPathProperty == null || backendProperty == null || maxNumTokensProperty == null || maxNumImagesProperty == null || enableSpeculativeDecodingProperty == null)
                {
                    throw new InvalidOperationException($"{nameof(LiteRtLmAndroidSmokeTestRunner)} does not expose required serialized build settings.");
                }

                modelPathProperty.stringValue = modelFileName;
                backendProperty.stringValue = backend;
                maxNumTokensProperty.intValue = Math.Max(1, maxNumTokens);
                maxNumImagesProperty.intValue = Math.Max(0, maxNumImages);
                enableSpeculativeDecodingProperty.boolValue = enableSpeculativeDecoding;
                if (runStandaloneBenchmarkProperty != null)
                {
                    runStandaloneBenchmarkProperty.boolValue = true;
                }

                if (benchmarkPrefillTokensProperty != null)
                {
                    benchmarkPrefillTokensProperty.intValue = Math.Max(1, benchmarkPrefillTokens);
                }

                if (benchmarkRunsProperty != null)
                {
                    benchmarkRunsProperty.intValue = Math.Max(3, benchmarkRunsProperty.intValue);
                }

                serializedRunner.ApplyModifiedPropertiesWithoutUndo();

                var sceneDirectory = Path.GetDirectoryName(Path.Combine(GetProjectRoot(), AndroidSmokeBuildScenePath));
                if (!string.IsNullOrWhiteSpace(sceneDirectory))
                {
                    Directory.CreateDirectory(sceneDirectory);
                }

                if (!EditorSceneManager.SaveScene(scene, AndroidSmokeBuildScenePath))
                {
                    throw new InvalidOperationException($"Failed to save Android smoke test build scene: {AndroidSmokeBuildScenePath}");
                }

                EditorSceneManager.CloseScene(scene, true);
                AssetDatabase.ImportAsset(AndroidSmokeBuildScenePath, ImportAssetOptions.ForceUpdate);
                Debug.Log($"LiteRT-LM Android smoke build scene generated: {AndroidSmokeBuildScenePath}, model={modelFileName}, backend={backend}, speculative={enableSpeculativeDecoding}, maxNumTokens={maxNumTokens}, maxNumImages={maxNumImages}");
                return AndroidSmokeBuildScenePath;
            }
            catch
            {
                if (scene.IsValid() && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }

                DeleteGeneratedAndroidSmokeBuildScene(AndroidSmokeBuildScenePath);
                throw;
            }
        }

        private static void DeleteGeneratedAndroidSmokeBuildScene(string scenePath)
        {
            if (string.IsNullOrWhiteSpace(scenePath) ||
                !string.Equals(scenePath, AndroidSmokeBuildScenePath, StringComparison.Ordinal))
            {
                return;
            }

            var absolutePath = Path.Combine(GetProjectRoot(), scenePath);
            if (File.Exists(absolutePath) || File.Exists(absolutePath + ".meta"))
            {
                AssetDatabase.DeleteAsset(scenePath);
            }
        }

        private static (string ModelPath, string Backend) SetAndroidSmokeTestSceneSettings(string modelFileName, string backend, bool enableSpeculativeDecoding)
        {
            var scene = EditorSceneManager.OpenScene(AndroidSmokeTestScenePath, OpenSceneMode.Single);
            var runner = UnityEngine.Object.FindAnyObjectByType<LiteRtLmAndroidSmokeTestRunner>();
            if (runner == null)
            {
                throw new InvalidOperationException($"Android smoke test scene does not contain {nameof(LiteRtLmAndroidSmokeTestRunner)}.");
            }

            var serializedRunner = new SerializedObject(runner);
            var modelPathProperty = serializedRunner.FindProperty("modelPath");
            var backendProperty = serializedRunner.FindProperty("backend");
            var enableSpeculativeDecodingProperty = serializedRunner.FindProperty("enableSpeculativeDecoding");
            var runStandaloneBenchmarkProperty = serializedRunner.FindProperty("runStandaloneBenchmark");
            var benchmarkRunsProperty = serializedRunner.FindProperty("benchmarkRuns");
            if (modelPathProperty == null)
            {
                throw new InvalidOperationException($"{nameof(LiteRtLmAndroidSmokeTestRunner)} does not expose serialized modelPath.");
            }

            if (backendProperty == null)
            {
                throw new InvalidOperationException($"{nameof(LiteRtLmAndroidSmokeTestRunner)} does not expose serialized backend.");
            }

            if (enableSpeculativeDecodingProperty == null)
            {
                throw new InvalidOperationException($"{nameof(LiteRtLmAndroidSmokeTestRunner)} does not expose serialized enableSpeculativeDecoding.");
            }

            var previousModelPath = modelPathProperty.stringValue;
            var previousBackend = backendProperty.stringValue;
            var changed = false;
            if (string.Equals(previousModelPath, modelFileName, StringComparison.Ordinal) &&
                string.Equals(previousBackend, backend, StringComparison.OrdinalIgnoreCase) &&
                enableSpeculativeDecodingProperty.boolValue == enableSpeculativeDecoding)
            {
                if (runStandaloneBenchmarkProperty == null ||
                    runStandaloneBenchmarkProperty.boolValue &&
                    (benchmarkRunsProperty == null || benchmarkRunsProperty.intValue >= 3))
                {
                    return (previousModelPath, previousBackend);
                }
            }

            modelPathProperty.stringValue = modelFileName;
            backendProperty.stringValue = backend;
            enableSpeculativeDecodingProperty.boolValue = enableSpeculativeDecoding;
            changed = true;
            if (runStandaloneBenchmarkProperty != null && !runStandaloneBenchmarkProperty.boolValue)
            {
                runStandaloneBenchmarkProperty.boolValue = true;
                changed = true;
            }

            if (benchmarkRunsProperty != null && benchmarkRunsProperty.intValue < 3)
            {
                benchmarkRunsProperty.intValue = 3;
                changed = true;
            }

            if (!changed)
            {
                return (previousModelPath, previousBackend);
            }

            serializedRunner.ApplyModifiedPropertiesWithoutUndo();
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene, AndroidSmokeTestScenePath))
            {
                throw new InvalidOperationException($"Failed to save Android smoke test scene: {AndroidSmokeTestScenePath}");
            }

            AssetDatabase.ImportAsset(AndroidSmokeTestScenePath, ImportAssetOptions.ForceUpdate);
            AssetDatabase.Refresh();
            return (previousModelPath, previousBackend);
        }

        private static void WithAndroidSmokeStreamingAssets(string modelFileName, Action buildAction)
        {
            var projectRoot = GetProjectRoot();
            var streamingAssetsDirectory = Path.Combine(projectRoot, "Assets", "StreamingAssets");
            var stashDirectory = Path.Combine(
                projectRoot,
                "Builds",
                "Android",
                "StreamingAssetsStash",
                DateTime.UtcNow.ToString("yyyyMMdd-HHmmssfff"));
            var movedFiles = new List<(string OriginalPath, string StashPath)>();

            try
            {
                if (Directory.Exists(streamingAssetsDirectory))
                {
                    foreach (var filePath in Directory.EnumerateFiles(streamingAssetsDirectory, "*", SearchOption.TopDirectoryOnly))
                    {
                        var fileName = Path.GetFileName(filePath);
                        if (IsAndroidSmokeStreamingAsset(fileName, modelFileName))
                        {
                            continue;
                        }

                        Directory.CreateDirectory(stashDirectory);
                        var stashPath = Path.Combine(stashDirectory, fileName);
                        File.Move(filePath, stashPath);
                        movedFiles.Add((filePath, stashPath));
                    }
                }

                AssetDatabase.Refresh();
                Debug.Log($"LiteRT-LM Android AVD smoke build staged StreamingAssets. StashedFiles={movedFiles.Count}");
                buildAction();
            }
            finally
            {
                for (var i = movedFiles.Count - 1; i >= 0; i--)
                {
                    var movedFile = movedFiles[i];
                    var originalDirectory = Path.GetDirectoryName(movedFile.OriginalPath);
                    if (!string.IsNullOrWhiteSpace(originalDirectory))
                    {
                        Directory.CreateDirectory(originalDirectory);
                    }

                    if (File.Exists(movedFile.OriginalPath))
                    {
                        File.Delete(movedFile.OriginalPath);
                    }

                    if (File.Exists(movedFile.StashPath))
                    {
                        File.Move(movedFile.StashPath, movedFile.OriginalPath);
                    }
                }

                AssetDatabase.Refresh();
            }
        }

        private static bool IsAndroidSmokeStreamingAsset(string fileName, string modelFileName)
        {
            return string.Equals(fileName, modelFileName, StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(fileName, modelFileName + ".meta", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(fileName, "README.txt", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(fileName, "README.txt.meta", StringComparison.OrdinalIgnoreCase);
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

        private static void ResetFunctionCallingBenchmarkStatus()
        {
            var statusPath = GetFunctionCallingBenchmarkStatusPath();
            var statusDirectory = Path.GetDirectoryName(statusPath);
            if (!string.IsNullOrWhiteSpace(statusDirectory))
            {
                Directory.CreateDirectory(statusDirectory);
            }

            using (var stream = new FileStream(statusPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite))
            {
            }
            WriteFunctionCallingBenchmarkStatus("START", DateTime.UtcNow.ToString("O"));
        }

        private static void WriteFunctionCallingBenchmarkStatus(string phase, string message)
        {
            var statusPath = GetFunctionCallingBenchmarkStatusPath();
            var line = $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}";
            Debug.Log($"[LiteRT-LM FunctionCallingBenchmark] {phase}: {message}");
            using var stream = new FileStream(statusPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
            using var writer = new StreamWriter(stream, Encoding.UTF8);
            writer.Write(line);
        }

        private static string GetFunctionCallingBenchmarkStatusPath()
        {
            return Path.Combine(GetProjectRoot(), FunctionCallingBenchmarkStatusRelativePath);
        }

        private static void StartFunctionCallingBenchmarkConsoleMirror()
        {
            StopFunctionCallingBenchmarkConsoleMirror();
            functionCallingBenchmarkConsoleMirrorActive = true;
            functionCallingBenchmarkConsoleMirrorPosition = 0;
            functionCallingBenchmarkNextConsoleMirrorTime = 0;
            EditorApplication.update += MirrorFunctionCallingBenchmarkStatusToConsole;
        }

        private static void StopFunctionCallingBenchmarkConsoleMirror()
        {
            if (!functionCallingBenchmarkConsoleMirrorActive)
            {
                return;
            }

            functionCallingBenchmarkConsoleMirrorActive = false;
            EditorApplication.update -= MirrorFunctionCallingBenchmarkStatusToConsole;
        }

        private static void MirrorFunctionCallingBenchmarkStatusToConsole()
        {
            if (!functionCallingBenchmarkConsoleMirrorActive ||
                EditorApplication.timeSinceStartup < functionCallingBenchmarkNextConsoleMirrorTime)
            {
                return;
            }

            functionCallingBenchmarkNextConsoleMirrorTime =
                EditorApplication.timeSinceStartup + FunctionCallingBenchmarkConsoleMirrorIntervalSeconds;

            var statusPath = GetFunctionCallingBenchmarkStatusPath();
            if (!File.Exists(statusPath))
            {
                return;
            }

            string newText;
            using (var stream = new FileStream(statusPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                if (functionCallingBenchmarkConsoleMirrorPosition > stream.Length)
                {
                    functionCallingBenchmarkConsoleMirrorPosition = 0;
                }

                stream.Seek(functionCallingBenchmarkConsoleMirrorPosition, SeekOrigin.Begin);
                using (var reader = new StreamReader(stream, Encoding.UTF8, true, 1024, true))
                {
                    newText = reader.ReadToEnd();
                }

                functionCallingBenchmarkConsoleMirrorPosition = stream.Position;
            }

            if (string.IsNullOrWhiteSpace(newText))
            {
                return;
            }

            foreach (var line in newText.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries))
            {
                Debug.Log($"[LiteRT-LM FunctionCallingBenchmark Progress] {line}");
            }

            if (newText.Contains("SUCCESS:") || newText.Contains("FAILURE:"))
            {
                StopFunctionCallingBenchmarkConsoleMirror();
                Debug.Log($"LiteRT-LM function-calling benchmark finished. Status={statusPath}");
            }
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
