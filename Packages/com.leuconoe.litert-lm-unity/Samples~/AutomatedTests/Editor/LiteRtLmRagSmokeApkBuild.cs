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
    /// Generates the RAG sample scene and builds its APK: no StreamingAssets (the index folder and gemma-4-E2B are
    /// pushed by Tools/Windows/Tests/Run-LiteRtLmAndroidRagSmokeTest.ps1) and its own application id, so it
    /// installs next to the other sample APKs.
    /// </summary>
    public static class LiteRtLmRagSmokeApkBuild
    {
        public const string ApplicationId = "com.Leuconoe.LiteRTLMUnity.rag";

        public static string RagSmokeScenePath => LiteRtLmSamplePaths.AutomatedScene("LiteRtLmRagSmokeTestScene");

        [MenuItem("LiteRT-LM/Scenes/Generate/RAG Smoke Test Scene (device)")]
        public static void GenerateRagSmokeTestScene()
        {
            var path = RagSmokeScenePath;
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

                var runnerObject = new GameObject("LiteRtLmRagSmokeTestRunner");
                SceneManager.MoveGameObjectToScene(runnerObject, scene);
                runnerObject.AddComponent<LiteRtLmRagSmokeTestRunner>();

                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".");
                if (!EditorSceneManager.SaveScene(scene, path))
                {
                    throw new InvalidOperationException($"Failed to save RAG smoke scene: {path}");
                }

                EditorSceneManager.CloseScene(scene, true);
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
                Debug.Log($"LiteRT-LM RAG smoke scene generated: {path}");
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

        [MenuItem("LiteRT-LM/Build/Android/Build RAG Smoke Test APK")]
        public static void BuildRagSmokeTestApk()
        {
            Build(Path.Combine(Path.GetDirectoryName(Application.dataPath) ?? ".",
                "Builds", "Android", "LiteRtLmAndroidRagSmokeTest.apk"));
        }

        public static void Build(string outputPath)
        {
            GenerateRagSmokeTestScene(); // always regenerate: scenes are generated, never hand-edited
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
                        scenes = new[] { RagSmokeScenePath },
                        locationPathName = outputPath,
                        target = BuildTarget.Android,
                        options = BuildOptions.Development,
                    });
                    if (report.summary.result != BuildResult.Succeeded)
                    {
                        throw new InvalidOperationException($"RAG smoke APK build failed with result {report.summary.result}.");
                    }
                });
                Debug.Log($"LiteRT-LM RAG smoke APK built: {outputPath} (applicationId {ApplicationId})");
            }
            finally
            {
                PlayerSettings.SetApplicationIdentifier(target, previousId);
                PlayerSettings.Android.targetArchitectures = previousArchitectures;
                PlayerSettings.SetScriptingBackend(target, previousScripting);
                // BuildPlayer saved ProjectSettings with the RAG id; persist the restore too, or the next build ships the wrong id.
                AssetDatabase.SaveAssets();
            }
        }
    }
}
