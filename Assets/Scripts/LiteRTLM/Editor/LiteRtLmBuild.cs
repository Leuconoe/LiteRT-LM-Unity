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
        private const string AndroidSmokeBuildScenePath = "Assets/Scenes/LiteRtLmAndroidSmokeTestBuildScene.generated.unity";
        private const string AndroidAsrSmokeBuildScenePath = "Assets/Scenes/LiteRtLmAsrSmokeTestBuildScene.generated.unity";
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
        private const string AsrSmokeDefaultModelPath = "parakeet_tdt_0.6b_v3_5s_i8.tflite";
        private const string AsrSmokeDefaultAudioPath = "Tactical Evaluation Results Report - March 5, 2025.mp3";
        private const string AsrSmokeTokenizerJsonPath = "parakeet-tdt-0.6b-v3/tokenizer.json";

        private static bool functionCallingBenchmarkConsoleMirrorActive;
        private static long functionCallingBenchmarkConsoleMirrorPosition;
        private static double functionCallingBenchmarkNextConsoleMirrorTime;

        private readonly struct AndroidSmokeBuildSettings
        {
            public AndroidSmokeBuildSettings(
                string modelFileName,
                string backend,
                string outputFileName,
                bool enableSpeculativeDecoding,
                int maxNumTokens = 64,
                int maxNumImages = 0,
                int benchmarkPrefillTokens = 64,
                bool packageModel = true)
            {
                ModelFileName = modelFileName;
                Backend = backend;
                OutputFileName = outputFileName;
                EnableSpeculativeDecoding = enableSpeculativeDecoding;
                MaxNumTokens = maxNumTokens;
                MaxNumImages = maxNumImages;
                BenchmarkPrefillTokens = benchmarkPrefillTokens;
                PackageModel = packageModel;
            }

            public string ModelFileName { get; }
            public string Backend { get; }
            public string OutputFileName { get; }
            public bool EnableSpeculativeDecoding { get; }
            public int MaxNumTokens { get; }
            public int MaxNumImages { get; }
            public int BenchmarkPrefillTokens { get; }
            public bool PackageModel { get; }
        }

        private readonly struct AndroidAsrSmokeBuildSettings
        {
            public AndroidAsrSmokeBuildSettings(
                string modelFileName,
                string audioFileName,
                string tokenizerJsonPath,
                string backend,
                string asrMode,
                string asrLanguage,
                string outputFileName)
            {
                ModelFileName = modelFileName;
                AudioFileName = audioFileName;
                TokenizerJsonPath = tokenizerJsonPath;
                Backend = backend;
                AsrMode = asrMode;
                AsrLanguage = asrLanguage;
                OutputFileName = outputFileName;
            }

            public string ModelFileName { get; }
            public string AudioFileName { get; }
            public string TokenizerJsonPath { get; }
            public string Backend { get; }
            public string AsrMode { get; }
            public string AsrLanguage { get; }
            public string OutputFileName { get; }
        }

        private static AndroidSmokeBuildSettings AndroidSmokeSettings(
            string modelFileName,
            string backend,
            string outputFileName,
            bool enableSpeculativeDecoding,
            int maxNumTokens = 64,
            int maxNumImages = 0,
            int benchmarkPrefillTokens = 64,
            bool packageModel = true)
        {
            return new AndroidSmokeBuildSettings(
                modelFileName,
                backend,
                outputFileName,
                enableSpeculativeDecoding,
                maxNumTokens,
                maxNumImages,
                benchmarkPrefillTokens,
                packageModel);
        }

        private static string GetCommandLineValue(string[] args, string key, string defaultValue)
        {
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (string.Equals(args[i], key, StringComparison.OrdinalIgnoreCase))
                {
                    return args[i + 1];
                }
            }

            return defaultValue;
        }

        private static int GetCommandLineInt(string[] args, string key, int defaultValue)
        {
            var value = GetCommandLineValue(args, key, string.Empty);
            return int.TryParse(value, out var parsed) ? parsed : defaultValue;
        }

        private static bool GetCommandLineBool(string[] args, string key, bool defaultValue)
        {
            var value = GetCommandLineValue(args, key, string.Empty);
            return bool.TryParse(value, out var parsed) ? parsed : defaultValue;
        }

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
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings("model.litertlm", "GPU", "LiteRtLmAndroidSmokeTest.apk", false));
        }

        public static void BuildAndroidAvdSmokeTestApkFromCommandLine()
        {
            var args = Environment.GetCommandLineArgs();
            var modelFileName = GetCommandLineValue(args, "-litertlmModel", string.Empty);
            var backend = GetCommandLineValue(args, "-litertlmBackend", "GPU");
            var outputFileName = GetCommandLineValue(args, "-litertlmOutputApk", string.Empty);
            var enableSpeculativeDecoding = GetCommandLineBool(args, "-litertlmSpeculative", false);
            var maxNumTokens = GetCommandLineInt(args, "-litertlmMaxNumTokens", 64);
            var maxNumImages = GetCommandLineInt(args, "-litertlmMaxNumImages", 0);
            var benchmarkPrefillTokens = GetCommandLineInt(args, "-litertlmBenchmarkPrefillTokens", 64);
            var packageModel = GetCommandLineBool(args, "-litertlmPackageModel", true);

            if (string.IsNullOrWhiteSpace(modelFileName))
            {
                throw new ArgumentException("-litertlmModel is required.");
            }

            if (string.IsNullOrWhiteSpace(outputFileName))
            {
                outputFileName = $"LiteRtLmAndroidSmokeTest-{Path.GetFileNameWithoutExtension(modelFileName)}-{backend}.apk";
            }

            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(
                modelFileName,
                backend,
                outputFileName,
                enableSpeculativeDecoding,
                maxNumTokens,
                maxNumImages,
                benchmarkPrefillTokens,
                packageModel));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma-4-E2B-it")]
        public static void BuildAndroidAvdSmokeTestApkGemma4()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingGemma4ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it.apk", true, 4000, 0, 128));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke APK/Gemma 4 E2B IT GPU No Speculative")]
        public static void BuildAndroidAvdSmokeTestApkGemma4NoSpeculative()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingGemma4ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-nospec.apk", false, 4000, 0, 128));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke APK/Gemma 4 E2B IT CPU")]
        public static void BuildAndroidAvdSmokeTestApkGemma4Cpu()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingGemma4ModelPath, "CPU", "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-CPU.apk", false, 4000, 0, 128));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma3-1b-it-int4")]
        public static void BuildAndroidAvdSmokeTestApkGemma1B()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingGemma1BModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma3-1b-it-int4 CPU")]
        public static void BuildAndroidAvdSmokeTestApkGemma1BCpu()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingGemma1BModelPath, "CPU", "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-CPU.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/gemma3-270m-it-q8")]
        public static void BuildAndroidAvdSmokeTestApkGemma270M()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingGemma270MModelPath, "GPU", "LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/mobile_actions_q8_ekv1024")]
        public static void BuildAndroidAvdSmokeTestApkMobileActions()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingMobileActionsModelPath, "GPU", "LiteRtLmAndroidSmokeTest-mobile_actions_q8_ekv1024.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen3-0.6B")]
        public static void BuildAndroidAvdSmokeTestApkQwen3()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingQwen3ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-Qwen3-0.6B.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-0.5B-Instruct")]
        public static void BuildAndroidAvdSmokeTestApkQwen25()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingQwen25ModelPath, "GPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-0.5B-Instruct CPU")]
        public static void BuildAndroidAvdSmokeTestApkQwen25Cpu()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingQwen25ModelPath, "CPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-1.5B-Instruct")]
        public static void BuildAndroidAvdSmokeTestApkQwen25_1_5B()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingQwen25_1_5BModelPath, "GPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build AVD Smoke Test APK/Qwen2.5-1.5B-Instruct CPU")]
        public static void BuildAndroidAvdSmokeTestApkQwen25_1_5BCpu()
        {
            BuildAndroidAvdSmokeTestApk(AndroidSmokeSettings(FunctionCallingQwen25_1_5BModelPath, "CPU", "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct-CPU.apk", false));
        }

        [MenuItem("LiteRT-LM/Android/Build ASR Smoke Test APK/Parakeet TFLite Inspect")]
        public static void BuildAndroidAsrSmokeTestApk()
        {
            BuildAndroidAsrSmokeTestApk(new AndroidAsrSmokeBuildSettings(
                AsrSmokeDefaultModelPath,
                AsrSmokeDefaultAudioPath,
                AsrSmokeTokenizerJsonPath,
                "GPU_FP16",
                "parakeet",
                "auto",
                "LiteRtLmAndroidAsrSmokeTest-parakeet-tdt-0.6b-v3.apk"));
        }

        public static void BuildAndroidAsrSmokeTestApkFromCommandLine()
        {
            var args = Environment.GetCommandLineArgs();
            var modelFileName = GetCommandLineValue(args, "-litertlmAsrModel", AsrSmokeDefaultModelPath);
            var audioFileName = GetCommandLineValue(args, "-litertlmAsrAudio", AsrSmokeDefaultAudioPath);
            var tokenizerJsonPath = GetCommandLineValue(args, "-litertlmAsrTokenizer", AsrSmokeTokenizerJsonPath);
            var backend = GetCommandLineValue(args, "-litertlmBackend", "GPU_FP16");
            var asrMode = GetCommandLineValue(args, "-litertlmAsrMode", "parakeet");
            var asrLanguage = GetCommandLineValue(args, "-litertlmAsrLanguage", "auto");
            var outputFileName = GetCommandLineValue(args, "-litertlmOutputApk", "LiteRtLmAndroidAsrSmokeTest-parakeet-tdt-0.6b-v3.apk");

            BuildAndroidAsrSmokeTestApk(new AndroidAsrSmokeBuildSettings(modelFileName, audioFileName, tokenizerJsonPath, backend, asrMode, asrLanguage, outputFileName));
        }

        private static void BuildAndroidAvdSmokeTestApk(AndroidSmokeBuildSettings settings)
        {
            if (string.IsNullOrWhiteSpace(settings.ModelFileName))
            {
                throw new ArgumentException("Android smoke test model file name is required.", nameof(settings));
            }

            if (string.IsNullOrWhiteSpace(settings.Backend))
            {
                throw new ArgumentException("Android smoke test backend is required.", nameof(settings));
            }

            var projectRoot = GetProjectRoot();
            var modelAssetPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", settings.ModelFileName);
            if (settings.PackageModel && !File.Exists(modelAssetPath))
            {
                throw new FileNotFoundException($"Android smoke test model not found: {modelAssetPath}", modelAssetPath);
            }

            var outputDirectory = Path.Combine(projectRoot, "Builds", "Android");
            Directory.CreateDirectory(outputDirectory);
            var outputPath = Path.Combine(outputDirectory, settings.OutputFileName);
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }

            var previousArchitectures = PlayerSettings.Android.targetArchitectures;
            var androidBuildTarget = UnityEditor.Build.NamedBuildTarget.Android;
            var previousScriptingBackend = PlayerSettings.GetScriptingBackend(androidBuildTarget);
            var previousUseDefaultGraphicsApis = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android);
            var previousGraphicsApis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
            var buildScenePath = CreateAndroidSmokeBuildScene(settings);

            try
            {
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, ScriptingImplementation.IL2CPP);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });

                WithAndroidSmokeStreamingAssets(settings.ModelFileName, settings.PackageModel, () =>
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

                Debug.Log($"LiteRT-LM Android AVD smoke APK built successfully: {outputPath}, model={settings.ModelFileName}, backend={settings.Backend}, packageModel={settings.PackageModel}, speculative={settings.EnableSpeculativeDecoding}, maxNumTokens={settings.MaxNumTokens}, maxNumImages={settings.MaxNumImages}");
            }
            finally
            {
                DeleteGeneratedBuildScene(buildScenePath);
                PlayerSettings.Android.targetArchitectures = previousArchitectures;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, previousScriptingBackend);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, previousUseDefaultGraphicsApis);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, previousGraphicsApis);
            }
        }

        private static void BuildAndroidAsrSmokeTestApk(AndroidAsrSmokeBuildSettings settings)
        {
            if (string.IsNullOrWhiteSpace(settings.ModelFileName))
            {
                throw new ArgumentException("Android ASR smoke test model file name is required.", nameof(settings));
            }

            if (string.IsNullOrWhiteSpace(settings.AudioFileName))
            {
                throw new ArgumentException("Android ASR smoke test audio file name is required.", nameof(settings));
            }
            if (string.IsNullOrWhiteSpace(settings.Backend))
            {
                throw new ArgumentException("Android ASR smoke test backend is required.", nameof(settings));
            }
            if (string.IsNullOrWhiteSpace(settings.TokenizerJsonPath))
            {
                throw new ArgumentException("Android ASR smoke test tokenizer path is required.", nameof(settings));
            }
            if (string.IsNullOrWhiteSpace(settings.AsrMode))
            {
                throw new ArgumentException("Android ASR smoke test mode is required.", nameof(settings));
            }

            var projectRoot = GetProjectRoot();
            var modelAssetPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", settings.ModelFileName);
            if (!File.Exists(modelAssetPath))
            {
                throw new FileNotFoundException($"Android ASR smoke test model not found: {modelAssetPath}", modelAssetPath);
            }

            var audioAssetPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", settings.AudioFileName);
            if (!File.Exists(audioAssetPath))
            {
                throw new FileNotFoundException($"Android ASR smoke test audio not found: {audioAssetPath}", audioAssetPath);
            }

            var tokenizerAssetPath = Path.Combine(projectRoot, "Assets", "StreamingAssets", settings.TokenizerJsonPath);
            if (!File.Exists(tokenizerAssetPath))
            {
                throw new FileNotFoundException($"Android ASR smoke test tokenizer not found: {tokenizerAssetPath}", tokenizerAssetPath);
            }

            var outputDirectory = Path.Combine(projectRoot, "Builds", "Android");
            Directory.CreateDirectory(outputDirectory);
            var outputPath = Path.Combine(outputDirectory, settings.OutputFileName);
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }

            var previousArchitectures = PlayerSettings.Android.targetArchitectures;
            var androidBuildTarget = UnityEditor.Build.NamedBuildTarget.Android;
            var previousScriptingBackend = PlayerSettings.GetScriptingBackend(androidBuildTarget);
            var previousUseDefaultGraphicsApis = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android);
            var previousGraphicsApis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
            var buildScenePath = CreateAndroidAsrSmokeBuildScene(settings);

            try
            {
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, ScriptingImplementation.IL2CPP);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });

                var packagedFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    settings.ModelFileName,
                    settings.ModelFileName + ".meta",
                    settings.AudioFileName,
                    settings.AudioFileName + ".meta",
                    settings.TokenizerJsonPath,
                    settings.TokenizerJsonPath + ".meta",
                };
                var encoderCompanionModel = GetWhisperEncoderCompanionModelFileName(settings.ModelFileName);
                if (!string.IsNullOrWhiteSpace(encoderCompanionModel) &&
                    File.Exists(Path.Combine(projectRoot, "Assets", "StreamingAssets", encoderCompanionModel)))
                {
                    packagedFiles.Add(encoderCompanionModel);
                    packagedFiles.Add(encoderCompanionModel + ".meta");
                }

                WithAndroidSelectedStreamingAssets(packagedFiles, () =>
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
                            $"Android ASR smoke build failed with result {report.summary.result}. See Unity Editor log for details.");
                    }
                });

                Debug.Log($"LiteRT-LM Android ASR smoke APK built successfully: {outputPath}, mode={settings.AsrMode}, model={settings.ModelFileName}, audio={settings.AudioFileName}, tokenizer={settings.TokenizerJsonPath}, backend={settings.Backend}");
            }
            finally
            {
                DeleteGeneratedBuildScene(buildScenePath);
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

        private static string GetWhisperEncoderCompanionModelFileName(string modelFileName)
        {
            if (string.IsNullOrWhiteSpace(modelFileName))
            {
                return string.Empty;
            }

            const string tfliteSuffix = ".tflite";
            const string f32Suffix = "_f32.tflite";
            if (modelFileName.EndsWith(f32Suffix, StringComparison.OrdinalIgnoreCase))
            {
                var preferred = modelFileName.Substring(0, modelFileName.Length - tfliteSuffix.Length) + "_encoder.tflite";
                var legacy = modelFileName.Substring(0, modelFileName.Length - f32Suffix.Length) + "_encoder_f32.tflite";
                var streamingAssetsRoot = Path.Combine(GetProjectRoot(), "Assets", "StreamingAssets");
                if (File.Exists(Path.Combine(streamingAssetsRoot, preferred)))
                {
                    return preferred;
                }
                if (File.Exists(Path.Combine(streamingAssetsRoot, legacy)))
                {
                    return legacy;
                }
                return preferred;
            }

            if (modelFileName.EndsWith(tfliteSuffix, StringComparison.OrdinalIgnoreCase))
            {
                return modelFileName.Substring(0, modelFileName.Length - tfliteSuffix.Length) + "_encoder.tflite";
            }

            return string.Empty;
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

        private static string CreateAndroidSmokeBuildScene(AndroidSmokeBuildSettings settings)
        {
            DeleteGeneratedBuildScene(AndroidSmokeBuildScenePath);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, GetTemporarySceneMode());
            try
            {
                var runnerObject = new GameObject("LiteRtLmAndroidSmokeTestRunner");
                SceneManager.MoveGameObjectToScene(runnerObject, scene);
                var runner = runnerObject.AddComponent<LiteRtLmAndroidSmokeTestRunner>();
                ApplyAndroidSmokeBuildSettings(runner, settings, runStandaloneBenchmark: true, minimumBenchmarkRuns: 3);

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
                Debug.Log($"LiteRT-LM Android smoke build scene generated: {AndroidSmokeBuildScenePath}, model={settings.ModelFileName}, backend={settings.Backend}, speculative={settings.EnableSpeculativeDecoding}, maxNumTokens={settings.MaxNumTokens}, maxNumImages={settings.MaxNumImages}");
                return AndroidSmokeBuildScenePath;
            }
            catch
            {
                if (scene.IsValid() && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }

                DeleteGeneratedBuildScene(AndroidSmokeBuildScenePath);
                throw;
            }
        }

        private static string CreateAndroidAsrSmokeBuildScene(AndroidAsrSmokeBuildSettings settings)
        {
            DeleteGeneratedBuildScene(AndroidAsrSmokeBuildScenePath);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, GetTemporarySceneMode());
            try
            {
                var runnerObject = new GameObject("LiteRtLmAsrSmokeTestRunner");
                SceneManager.MoveGameObjectToScene(runnerObject, scene);
                var runner = runnerObject.AddComponent<LiteRtLmAsrSmokeTestRunner>();
                ApplyAndroidAsrSmokeBuildSettings(runner, settings);

                var sceneDirectory = Path.GetDirectoryName(Path.Combine(GetProjectRoot(), AndroidAsrSmokeBuildScenePath));
                if (!string.IsNullOrWhiteSpace(sceneDirectory))
                {
                    Directory.CreateDirectory(sceneDirectory);
                }

                if (!EditorSceneManager.SaveScene(scene, AndroidAsrSmokeBuildScenePath))
                {
                    throw new InvalidOperationException($"Failed to save Android ASR smoke test build scene: {AndroidAsrSmokeBuildScenePath}");
                }

                EditorSceneManager.CloseScene(scene, true);
                AssetDatabase.ImportAsset(AndroidAsrSmokeBuildScenePath, ImportAssetOptions.ForceUpdate);
                Debug.Log($"LiteRT-LM Android ASR smoke build scene generated: {AndroidAsrSmokeBuildScenePath}, model={settings.ModelFileName}, audio={settings.AudioFileName}, backend={settings.Backend}");
                return AndroidAsrSmokeBuildScenePath;
            }
            catch
            {
                if (scene.IsValid() && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }

                DeleteGeneratedBuildScene(AndroidAsrSmokeBuildScenePath);
                throw;
            }
        }

        private static NewSceneMode GetTemporarySceneMode()
        {
            var activeScene = SceneManager.GetActiveScene();
            var hasSavedActiveScene = activeScene.IsValid() && !string.IsNullOrEmpty(activeScene.path);
            return Application.isBatchMode || !hasSavedActiveScene
                ? NewSceneMode.Single
                : NewSceneMode.Additive;
        }

        private static void ApplyAndroidSmokeBuildSettings(
            LiteRtLmAndroidSmokeTestRunner runner,
            AndroidSmokeBuildSettings settings,
            bool runStandaloneBenchmark,
            int minimumBenchmarkRuns)
        {
            var serializedRunner = new SerializedObject(runner);
            FindRequiredProperty(serializedRunner, "modelPath").stringValue = settings.ModelFileName;
            FindRequiredProperty(serializedRunner, "backend").stringValue = settings.Backend;
            FindRequiredProperty(serializedRunner, "maxNumTokens").intValue = Math.Max(1, settings.MaxNumTokens);
            FindRequiredProperty(serializedRunner, "maxNumImages").intValue = Math.Max(0, settings.MaxNumImages);
            FindRequiredProperty(serializedRunner, "enableSpeculativeDecoding").boolValue = settings.EnableSpeculativeDecoding;

            var runStandaloneBenchmarkProperty = serializedRunner.FindProperty("runStandaloneBenchmark");
            if (runStandaloneBenchmarkProperty != null)
            {
                runStandaloneBenchmarkProperty.boolValue = runStandaloneBenchmark;
            }

            var benchmarkPrefillTokensProperty = serializedRunner.FindProperty("benchmarkPrefillTokens");
            if (benchmarkPrefillTokensProperty != null)
            {
                benchmarkPrefillTokensProperty.intValue = Math.Max(1, settings.BenchmarkPrefillTokens);
            }

            var benchmarkRunsProperty = serializedRunner.FindProperty("benchmarkRuns");
            if (benchmarkRunsProperty != null && minimumBenchmarkRuns > 0)
            {
                benchmarkRunsProperty.intValue = Math.Max(minimumBenchmarkRuns, benchmarkRunsProperty.intValue);
            }

            serializedRunner.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void ApplyAndroidAsrSmokeBuildSettings(
            LiteRtLmAsrSmokeTestRunner runner,
            AndroidAsrSmokeBuildSettings settings)
        {
            var serializedRunner = new SerializedObject(runner);
            FindRequiredProperty(serializedRunner, "modelPath").stringValue = settings.ModelFileName;
            FindRequiredProperty(serializedRunner, "audioPath").stringValue = settings.AudioFileName;
            FindRequiredProperty(serializedRunner, "tokenizerJsonPath").stringValue = settings.TokenizerJsonPath;
            FindRequiredProperty(serializedRunner, "backend").stringValue = settings.Backend;
            FindRequiredProperty(serializedRunner, "asrMode").stringValue = settings.AsrMode;
            FindRequiredProperty(serializedRunner, "asrLanguage").stringValue = settings.AsrLanguage;
            serializedRunner.ApplyModifiedPropertiesWithoutUndo();
        }

        private static SerializedProperty FindRequiredProperty(SerializedObject serializedObject, string propertyName)
        {
            var property = serializedObject.FindProperty(propertyName);
            if (property == null)
            {
                throw new InvalidOperationException($"{serializedObject.targetObject.GetType().Name} does not expose serialized {propertyName}.");
            }

            return property;
        }

        private static void DeleteGeneratedBuildScene(string scenePath)
        {
            if (string.IsNullOrWhiteSpace(scenePath) ||
                (!string.Equals(scenePath, AndroidSmokeBuildScenePath, StringComparison.Ordinal) &&
                 !string.Equals(scenePath, AndroidAsrSmokeBuildScenePath, StringComparison.Ordinal)))
            {
                return;
            }

            var absolutePath = Path.Combine(GetProjectRoot(), scenePath);
            if (File.Exists(absolutePath) || File.Exists(absolutePath + ".meta"))
            {
                AssetDatabase.DeleteAsset(scenePath);
            }
        }

        private static void WithAndroidSmokeStreamingAssets(string modelFileName, bool packageModel, Action buildAction)
        {
            var allowedFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "README.txt",
                "README.txt.meta",
            };

            if (packageModel)
            {
                allowedFiles.Add(modelFileName);
                allowedFiles.Add(modelFileName + ".meta");
            }

            WithAndroidSelectedStreamingAssets(allowedFiles, buildAction);
        }

        private static void WithAndroidSelectedStreamingAssets(HashSet<string> allowedFiles, Action buildAction)
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
                        if (allowedFiles.Contains(fileName))
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
                Debug.Log($"LiteRT-LM Android build staged StreamingAssets. AllowedFiles={allowedFiles.Count}, StashedFiles={movedFiles.Count}");
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
