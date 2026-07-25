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
        public const string LlmChatTestScenePath = "Assets/Scenes/Tests/LiteRtLmLlmChatTestScene.unity";
        public const string AsrTestScenePath = "Assets/Scenes/Tests/LiteRtLmAsrTestScene.unity";
        public const string MultimodalTestScenePath = "Assets/Scenes/Tests/LiteRtLmMultimodalTestScene.unity";
        public const string AsrFunctionCallingTestScenePath = "Assets/Scenes/Tests/LiteRtLmAsrFunctionCallingTestScene.unity";
        public const string MultimodalFunctionCallingTestScenePath = "Assets/Scenes/Tests/LiteRtLmMultimodalFunctionCallingTestScene.unity";
        public const string TranslateTestScenePath = "Assets/Scenes/Tests/LiteRtLmTranslateTestScene.unity";

        private const string AsrFunctionCallingDefaultAsrModelPath = "ASR/whisper-tiny/whisper_tiny_30s_i8.tflite";
        private const string AsrFunctionCallingDefaultAudioPath = "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3";
        private const string AsrFunctionCallingDefaultTokenizerJsonPath = "ASR/whisper-tiny/tokenizer.json";
        private const string AsrFunctionCallingStatusFileName = "LiteRtLmAsrFunctionCallingDemo.status.txt";
        private const string MultimodalFunctionCallingStatusFileName = "LiteRtLmMultimodalFunctionCallingDemo.status.txt";

        private static readonly string[] AllTestScenePaths =
        {
            LlmChatTestScenePath,
            AsrTestScenePath,
            MultimodalTestScenePath,
            AsrFunctionCallingTestScenePath,
            MultimodalFunctionCallingTestScenePath,
            TranslateTestScenePath,
        };

        [MenuItem("LiteRT-LM/Test Scenes/Generate All")]
        public static void GenerateAllTestScenes()
        {
            GenerateLlmChatTestScene();
            GenerateAsrTestScene();
            GenerateMultimodalTestScene();
            GenerateAsrFunctionCallingTestScene();
            GenerateMultimodalFunctionCallingTestScene();
            GenerateTranslateTestScene();
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

        [MenuItem("LiteRT-LM/Test Scenes/Generate LLM Chat Test Scene")]
        public static void GenerateLlmChatTestScene()
        {
            CreateTestScene(LlmChatTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmLlmChatTestRunner>(scene, "LiteRtLmLlmChatTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Test Scenes/Generate ASR Test Scene")]
        public static void GenerateAsrTestScene()
        {
            CreateTestScene(AsrTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmAsrTestRunner>(scene, "LiteRtLmAsrTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Test Scenes/Generate Multimodal Test Scene")]
        public static void GenerateMultimodalTestScene()
        {
            CreateTestScene(MultimodalTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmMultimodalTestRunner>(scene, "LiteRtLmMultimodalTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Test Scenes/Generate ASR Function Calling Test Scene")]
        public static void GenerateAsrFunctionCallingTestScene()
        {
            CreateTestScene(AsrFunctionCallingTestScenePath, scene =>
            {
                var runner = CreateRunnerObject<LiteRtLmAsrFunctionCallingDemoRunner>(scene, "LiteRtLmAsrFunctionCallingDemoRunner");
                var serializedRunner = new SerializedObject(runner);
                FindRequiredProperty(serializedRunner, "asrModelPath").stringValue = AsrFunctionCallingDefaultAsrModelPath;
                FindRequiredProperty(serializedRunner, "audioPath").stringValue = AsrFunctionCallingDefaultAudioPath;
                FindRequiredProperty(serializedRunner, "tokenizerJsonPath").stringValue = AsrFunctionCallingDefaultTokenizerJsonPath;
                FindRequiredProperty(serializedRunner, "asrBackend").stringValue = "CPU";
                FindRequiredProperty(serializedRunner, "asrLanguage").stringValue = "ko";
                FindRequiredProperty(serializedRunner, "llmModelPath").stringValue = "LLM/gemma3-1b/gemma3-1b-it-int4.litertlm";
                FindRequiredProperty(serializedRunner, "llmBackend").stringValue = "GPU";
                FindRequiredProperty(serializedRunner, "llmMaxNumTokens").intValue = 512;
                serializedRunner.ApplyModifiedPropertiesWithoutUndo();

                AddStatusHudOverlay(runner.gameObject, AsrFunctionCallingStatusFileName);
            });
        }

        [MenuItem("LiteRT-LM/Test Scenes/Generate Multimodal Function Calling Test Scene")]
        public static void GenerateMultimodalFunctionCallingTestScene()
        {
            CreateTestScene(MultimodalFunctionCallingTestScenePath, scene =>
            {
                var runner = CreateRunnerObject<LiteRtLmMultimodalFunctionCallingRunner>(scene, "LiteRtLmMultimodalFunctionCallingRunner");
                AddStatusHudOverlay(runner.gameObject, MultimodalFunctionCallingStatusFileName);
            });
        }

        [MenuItem("LiteRT-LM/Test Scenes/Generate Translate Test Scene")]
        public static void GenerateTranslateTestScene()
        {
            CreateTestScene(TranslateTestScenePath, scene =>
            {
                CreateRunnerObject<LiteRtLmTranslateTestRunner>(scene, "LiteRtLmTranslateTestRunner");
            });
        }

        [MenuItem("LiteRT-LM/Test Scenes/Register Test Scenes In Build Settings")]
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
                scenes.Add(new EditorBuildSettingsScene(testScenePath, false));
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
