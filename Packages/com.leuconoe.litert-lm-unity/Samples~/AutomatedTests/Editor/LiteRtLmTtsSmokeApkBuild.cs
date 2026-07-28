using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Builds the headless TTS smoke APK.
    ///
    /// Kept apart from <see cref="LiteRtLmBuild"/> because the StreamingAssets
    /// selection is the whole point of a per-test APK here: the project's
    /// StreamingAssets holds multi-GB LLM and ASR models, and packaging all of
    /// them would produce an APK too large to install. Only the Supertonic ladder
    /// goes in — 202 MB.
    /// </summary>
    public static class LiteRtLmTtsSmokeApkBuild
    {
        private const string TtsRoot = "TTS/supertonic-litert";

        [MenuItem("LiteRT-LM/Build/Android/Build TTS Smoke Test APK")]
        public static void BuildTtsSmokeTestApk()
        {
            Build(Path.Combine(
                Path.GetDirectoryName(Application.dataPath) ?? ".",
                "Builds", "AndroidBuilds", "LiteRtLmAndroidTtsSmokeTest.apk"));
        }

        /// <summary>Batch-mode entry point: <c>-executeMethod …BuildFromCommandLine</c>.</summary>
        public static void BuildFromCommandLine()
        {
            try
            {
                var output = ReadArgument("-litertlmOutputApk") ??
                             Path.Combine("Builds", "AndroidBuilds", "LiteRtLmAndroidTtsSmokeTest.apk");
                if (!Path.IsPathRooted(output))
                {
                    output = Path.Combine(Path.GetDirectoryName(Application.dataPath) ?? ".", output);
                }

                Build(output);
                EditorApplication.Exit(0);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorApplication.Exit(1);
            }
        }

        private static void Build(string outputPath)
        {
            var scenePath = LiteRtLmTtsSmokeSceneGenerator.TtsSmokeScenePath;
            if (!File.Exists(Path.Combine(
                    Path.GetDirectoryName(Application.dataPath) ?? ".", scenePath)))
            {
                LiteRtLmTtsSmokeSceneGenerator.GenerateTtsSmokeTestScene();
            }

            var projectRoot = Path.GetDirectoryName(Application.dataPath) ?? ".";
            var ttsRootAbsolute = Path.Combine(projectRoot, "Assets", "StreamingAssets", TtsRoot);
            if (!Directory.Exists(ttsRootAbsolute))
            {
                throw new DirectoryNotFoundException(
                    $"TTS model set not found: {ttsRootAbsolute}. " +
                    "Run Tools/Windows/Deploy-SupertonicLiteRt.ps1 first.");
            }

            // A folder entry is enough: the selection helper keeps whole
            // subtrees, so the ladder does not have to be enumerated here.
            var packaged = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                TtsRoot,
                TtsRoot + ".meta",
            };

            var bytes = 0L;
            foreach (var file in Directory.GetFiles(ttsRootAbsolute, "*.tflite", SearchOption.AllDirectories))
            {
                bytes += new FileInfo(file).Length;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");

            var androidBuildTarget = UnityEditor.Build.NamedBuildTarget.Android;
            var previousArchitectures = PlayerSettings.Android.targetArchitectures;
            var previousScripting = PlayerSettings.GetScriptingBackend(androidBuildTarget);
            var previousUseDefaultGraphics = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android);
            var previousGraphics = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);

            try
            {
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, ScriptingImplementation.IL2CPP);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });

                Debug.Log($"LiteRT-LM TTS smoke APK: packaging {TtsRoot} " +
                          $"({bytes / 1048576.0:0.0} MB of tflite) into {outputPath}");

                LiteRtLmBuild.BuildWithSelectedStreamingAssets(packaged, () =>
                {
                    var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                    {
                        scenes = new[] { scenePath },
                        locationPathName = outputPath,
                        target = BuildTarget.Android,
                        options = BuildOptions.Development,
                    });

                    if (report.summary.result != BuildResult.Succeeded)
                    {
                        throw new InvalidOperationException(
                            $"TTS smoke APK build failed with result {report.summary.result}.");
                    }
                });

                Debug.Log($"LiteRT-LM TTS smoke APK built: {outputPath}");
            }
            finally
            {
                PlayerSettings.Android.targetArchitectures = previousArchitectures;
                PlayerSettings.SetScriptingBackend(androidBuildTarget, previousScripting);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, previousUseDefaultGraphics);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, previousGraphics);
            }
        }

        private static string ReadArgument(string name)
        {
            var args = Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                {
                    return args[i + 1];
                }
            }

            return null;
        }
    }
}
