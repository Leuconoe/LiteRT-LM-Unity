using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Generates the headless TTS device scene. Kept separate from the interactive
    /// TTS scene generator because this one belongs in the Automated Tests sample:
    /// it is what goes into an APK to answer "does Supertonic run on kona, and how
    /// fast", and it writes a status file rather than drawing a UI.
    ///
    /// Scenes in this project are generated, never hand-authored, so that a change
    /// to a runner's serialized fields is picked up by re-running the menu item
    /// instead of by editing YAML.
    /// </summary>
    public static class LiteRtLmTtsSmokeSceneGenerator
    {
        public static string TtsSmokeScenePath =>
            LiteRtLmSamplePaths.AutomatedScene("LiteRtLmTtsSmokeTestScene");

        [MenuItem("LiteRT-LM/Scenes/Generate/TTS Smoke Test Scene (device)")]
        public static void GenerateTtsSmokeTestScene()
        {
            var path = TtsSmokeScenePath;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, GetTemporarySceneMode());
            try
            {
                // A camera and an AudioListener so the player does not warn; the
                // runner itself draws nothing and plays nothing — the WAVs are
                // pulled back and judged on the desktop.
                var cameraObject = new GameObject("Main Camera");
                SceneManager.MoveGameObjectToScene(cameraObject, scene);
                cameraObject.tag = "MainCamera";
                var camera = cameraObject.AddComponent<Camera>();
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = new Color(0.12f, 0.12f, 0.14f, 1f);
                cameraObject.AddComponent<AudioListener>();

                var runnerObject = new GameObject("LiteRtLmTtsSmokeTestRunner");
                SceneManager.MoveGameObjectToScene(runnerObject, scene);
                runnerObject.AddComponent<LiteRtLmTtsSmokeTestRunner>();

                var directory = Path.GetDirectoryName(Path.Combine(GetProjectRoot(), path));
                if (!string.IsNullOrWhiteSpace(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                if (!EditorSceneManager.SaveScene(scene, path))
                {
                    throw new InvalidOperationException($"Failed to save TTS smoke scene: {path}");
                }

                EditorSceneManager.CloseScene(scene, true);
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
                Debug.Log($"LiteRT-LM TTS smoke scene generated: {path}");
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

        private static NewSceneMode GetTemporarySceneMode()
        {
            var active = SceneManager.GetActiveScene();
            var hasSaved = active.IsValid() && !string.IsNullOrEmpty(active.path);
            return Application.isBatchMode || !hasSaved ? NewSceneMode.Single : NewSceneMode.Additive;
        }

        private static string GetProjectRoot()
        {
            var root = Directory.GetParent(Application.dataPath);
            if (root == null)
            {
                throw new InvalidOperationException("Failed to resolve Unity project root.");
            }

            return root.FullName;
        }
    }
}
