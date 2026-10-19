using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Generates the Jev sample scene and builds its APK.
    ///
    /// The APK packages no StreamingAssets (the 239 MB trunk, tokenizer and head are pushed by
    /// Tools/Windows/Tests/Run-LiteRtLmAndroidJevSmokeTest.ps1) and installs under its own
    /// application id, so it sits next to the FC prompt-set smoke APK instead of replacing it.
    /// </summary>
    public static class LiteRtLmJevSmokeApkBuild
    {
        public const string ApplicationId = "com.Leuconoe.LiteRTLMUnity.jev";

        public static string JevSmokeScenePath => LiteRtLmSamplePaths.AutomatedScene("LiteRtLmJevSmokeTestScene");

        [MenuItem("LiteRT-LM/Scenes/Generate/Jev Smoke Test Scene (device)")]
        public static void GenerateJevSmokeTestScene()
        {
            var path = JevSmokeScenePath;
            var active = SceneManager.GetActiveScene();
            var mode = Application.isBatchMode || !active.IsValid() || string.IsNullOrEmpty(active.path)
                ? NewSceneMode.Single
                : NewSceneMode.Additive;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, mode);
            try
            {
                var cameraObject = new GameObject("Main Camera");
                SceneManager.MoveGameObjectToScene(cameraObject, scene);
                cameraObject.tag = "MainCamera";
                var camera = cameraObject.AddComponent<Camera>();
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = new Color(0.12f, 0.12f, 0.14f, 1f);
                cameraObject.AddComponent<AudioListener>();

                var runnerObject = new GameObject("LiteRtLmJevSmokeTestRunner");
                SceneManager.MoveGameObjectToScene(runnerObject, scene);
                runnerObject.AddComponent<LiteRtLmJevSmokeTestRunner>();

                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
                if (!EditorSceneManager.SaveScene(scene, path))
                {
                    throw new InvalidOperationException($"Failed to save Jev smoke scene: {path}");
                }

                EditorSceneManager.CloseScene(scene, true);
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
                Debug.Log($"LiteRT-LM Jev smoke scene generated: {path}");
            }
            catch
            {
                if (scene.IsValid() && scene.isLoaded)
                {
                    EditorSceneManager.CloseScene(scene, true);
                }

                throw;
            }
        }

        [MenuItem("LiteRT-LM/Build/Android/Build Jev Smoke Test APK")]
        public static void BuildJevSmokeTestApk()
        {
            Build(Path.Combine(Path.GetDirectoryName(Application.dataPath) ?? ".",
                "Builds", "Android", "LiteRtLmAndroidJevSmokeTest.apk"));
        }

        public static void Build(string outputPath)
        {
            GenerateJevSmokeTestScene(); // always regenerate: scenes are generated, never hand-edited
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");

            var target = UnityEditor.Build.NamedBuildTarget.Android;
            var previousId = PlayerSettings.GetApplicationIdentifier(target);
            var previousArchitectures = PlayerSettings.Android.targetArchitectures;
            var previousScripting = PlayerSettings.GetScriptingBackend(target);
            try
            {
                PlayerSettings.SetApplicationIdentifier(target, ApplicationId);
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.SetScriptingBackend(target, ScriptingImplementation.IL2CPP);
                LiteRtLmBuild.BuildWithSelectedStreamingAssets(new HashSet<string>(), () =>
                {
                    var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                    {
                        scenes = new[] { JevSmokeScenePath },
                        locationPathName = outputPath,
                        target = BuildTarget.Android,
                        options = BuildOptions.Development,
                    });
                    if (report.summary.result != BuildResult.Succeeded)
                    {
                        throw new InvalidOperationException($"Jev smoke APK build failed with result {report.summary.result}.");
                    }
                });
                Debug.Log($"LiteRT-LM Jev smoke APK built: {outputPath} (applicationId {ApplicationId})");
            }
            finally
            {
                PlayerSettings.SetApplicationIdentifier(target, previousId);
                PlayerSettings.Android.targetArchitectures = previousArchitectures;
                PlayerSettings.SetScriptingBackend(target, previousScripting);
                // BuildPlayer saved ProjectSettings with the Jev id; persist the restore too, or the
                // next FC smoke APK silently builds as com.…LiteRTLMUnity.jev.
                AssetDatabase.SaveAssets();
            }
        }
    }
}
