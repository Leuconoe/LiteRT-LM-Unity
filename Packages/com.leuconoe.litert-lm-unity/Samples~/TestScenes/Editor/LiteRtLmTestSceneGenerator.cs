using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LiteRTLM.Unity.Editor
{
    public static class LiteRtLmTestSceneGenerator
    {
        // Resolved from where the sample was imported; see LiteRtLmSamplePaths.
        public static string LlmChatTestScenePath => LiteRtLmSamplePaths.Scene("LiteRtLmLlmChatTestScene");
        public static string AsrTestScenePath => LiteRtLmSamplePaths.Scene("LiteRtLmAsrTestScene");
        public static string MultimodalTestScenePath => LiteRtLmSamplePaths.Scene("LiteRtLmMultimodalTestScene");
        public static string AsrFunctionCallingTestScenePath => LiteRtLmSamplePaths.Scene("LiteRtLmAsrFunctionCallingTestScene");
        public static string MultimodalFunctionCallingTestScenePath => LiteRtLmSamplePaths.Scene("LiteRtLmMultimodalFunctionCallingTestScene");
        public static string TranslateTestScenePath => LiteRtLmSamplePaths.Scene("LiteRtLmTranslateTestScene");
        public static string TtsTestScenePath => LiteRtLmSamplePaths.Scene("LiteRtLmTtsTestScene");


        private static string[] AllTestScenePaths => new[]
        {
            LlmChatTestScenePath,
            AsrTestScenePath,
            MultimodalTestScenePath,
            AsrFunctionCallingTestScenePath,
            MultimodalFunctionCallingTestScenePath,
            TranslateTestScenePath,
            TtsTestScenePath,
        };

        [MenuItem("LiteRT-LM/Scenes/Generate/All Test Scenes")]
        public static void GenerateAllTestScenes()
        {
            GenerateLlmChatTestScene();
            GenerateAsrTestScene();
            GenerateMultimodalTestScene();
            GenerateAsrFunctionCallingTestScene();
            GenerateMultimodalFunctionCallingTestScene();
            GenerateTranslateTestScene();
            GenerateTtsTestScene();
            RegisterTestScenesInBuildSettings();
            Debug.Log($"LiteRT-LM test scene generation completed. Scenes={AllTestScenePaths.Length}");
        }

        public static void GenerateAllFromCommandLine()
        {
            try
            {
                GenerateAllTestScenes();
                Debug.Log("LiteRT-LM test scene generation succeeded (command line).");
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/LLM Chat Test Scene")]
        public static void GenerateLlmChatTestScene()
        {
            CreateTestScene(LlmChatTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmLlmChatTestRunner>(scene, "LiteRtLmLlmChatTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/ASR Test Scene")]
        public static void GenerateAsrTestScene()
        {
            CreateTestScene(AsrTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmAsrTestRunner>(scene, "LiteRtLmAsrTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/Multimodal Test Scene")]
        public static void GenerateMultimodalTestScene()
        {
            CreateTestScene(MultimodalTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmMultimodalTestRunner>(scene, "LiteRtLmMultimodalTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/ASR Function Calling Test Scene")]
        public static void GenerateAsrFunctionCallingTestScene()
        {
            CreateTestScene(AsrFunctionCallingTestScenePath, scene =>
            {
                // Interactive runner: the batch demo runner lives in the Automated
                // Tests sample and is used by the generated device build scenes.
                CreateRunnerObject<LiteRtLmAsrFunctionCallingTestRunner>(scene, "LiteRtLmAsrFunctionCallingTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/Multimodal Function Calling Test Scene")]
        public static void GenerateMultimodalFunctionCallingTestScene()
        {
            CreateTestScene(MultimodalFunctionCallingTestScenePath, scene =>
            {
                // Interactive runner; the batch demo lives in the Automated Tests sample.
                CreateRunnerObject<LiteRtLmMultimodalFunctionCallingTestRunner>(
                    scene, "LiteRtLmMultimodalFunctionCallingTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/Translate Test Scene")]
        public static void GenerateTranslateTestScene()
        {
            CreateTestScene(TranslateTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmTranslateTestRunner>(scene, "LiteRtLmTranslateTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/TTS Test Scene")]
        public static void GenerateTtsTestScene()
        {
            CreateTestScene(TtsTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmTtsTestRunner>(scene, "LiteRtLmTtsTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Scenes/Generate/Register Test Scenes In Build Settings")]
        public static void RegisterTestScenesInBuildSettings()
        {
            var projectRoot = GetProjectRoot();
            var scenes = new List<EditorBuildSettingsScene>();
            var knownPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var scene in EditorBuildSettings.scenes)
            {
                if (string.IsNullOrWhiteSpace(scene.path) ||
                    !File.Exists(Path.Combine(projectRoot, scene.path)))
                {
                    Debug.Log($"LiteRT-LM build settings pruning dead scene entry: {scene.path}");
                    continue;
                }

                if (!knownPaths.Add(scene.path))
                {
                    continue;
                }

                scenes.Add(scene);
            }

            foreach (var testScenePath in AllTestScenePaths)
            {
                if (knownPaths.Contains(testScenePath))
                {
                    continue;
                }

                if (!File.Exists(Path.Combine(projectRoot, testScenePath)))
                {
                    Debug.LogWarning($"LiteRT-LM build settings skipping missing test scene: {testScenePath}. Run the scene generator first.");
                    continue;
                }

                knownPaths.Add(testScenePath);
                // Enabled: these are the scenes the navigator walks through on a
                // device build. Added disabled they registered but never appeared.
                scenes.Add(new EditorBuildSettingsScene(testScenePath, true));
            }

            EditorBuildSettings.scenes = scenes.ToArray();
            Debug.Log($"LiteRT-LM build settings updated. SceneEntries={scenes.Count}");
        }

        private static void CreateTestScene(string scenePath, Action<Scene> populate)
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, GetTemporarySceneMode());
            try
            {
                CreateMainCamera(scene);
                CreateSceneNavigator(scene);
                populate(scene);

                var sceneDirectory = Path.GetDirectoryName(Path.Combine(GetProjectRoot(), scenePath));
                if (!string.IsNullOrWhiteSpace(sceneDirectory))
                {
                    Directory.CreateDirectory(sceneDirectory);
                }

                if (!EditorSceneManager.SaveScene(scene, scenePath))
                {
                    throw new InvalidOperationException($"Failed to save test scene: {scenePath}");
                }

                EditorSceneManager.CloseScene(scene, true);
                AssetDatabase.ImportAsset(scenePath, ImportAssetOptions.ForceUpdate);
                Debug.Log($"LiteRT-LM test scene generated: {scenePath}");
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

        private static void CreateMainCamera(Scene scene)
        {
            var cameraObject = new GameObject("Main Camera");
            SceneManager.MoveGameObjectToScene(cameraObject, scene);
            cameraObject.tag = "MainCamera";
            var camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.12f, 0.12f, 0.14f, 1f);
            cameraObject.AddComponent<AudioListener>();
            cameraObject.transform.position = new Vector3(0f, 1f, -10f);
        }

        /// <summary>
        /// Adds the shared prev/next bar so the generated scenes can be walked
        /// through on a device. It releases every loaded model before each load.
        /// </summary>
        private static void CreateSceneNavigator(Scene scene)
        {
            var navigatorObject = new GameObject("LiteRtLmSceneNavigator");
            SceneManager.MoveGameObjectToScene(navigatorObject, scene);
            navigatorObject.AddComponent<LiteRtLmSceneNavigator>();
        }

        private static TRunner CreateRunnerObject<TRunner>(Scene scene, string objectName)
            where TRunner : Component
        {
            var runnerObject = new GameObject(objectName);
            SceneManager.MoveGameObjectToScene(runnerObject, scene);
            return runnerObject.AddComponent<TRunner>();
        }

        private static void AddStatusHudOverlay(GameObject host, string statusFileName)
        {
            var overlay = host.AddComponent<LiteRtLmStatusHudOverlay>();
            var serializedOverlay = new SerializedObject(overlay);
            FindRequiredProperty(serializedOverlay, "statusFileName").stringValue = statusFileName;
            serializedOverlay.ApplyModifiedPropertiesWithoutUndo();
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

        private static NewSceneMode GetTemporarySceneMode()
        {
            var activeScene = SceneManager.GetActiveScene();
            var hasSavedActiveScene = activeScene.IsValid() && !string.IsNullOrEmpty(activeScene.path);
            return Application.isBatchMode || !hasSavedActiveScene
                ? NewSceneMode.Single
                : NewSceneMode.Additive;
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
