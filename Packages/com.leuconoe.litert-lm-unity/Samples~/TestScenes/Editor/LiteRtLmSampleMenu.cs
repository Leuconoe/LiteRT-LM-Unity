using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Housekeeping entries for the sample scenes.
    ///
    /// The prev/next bar loads scenes by name, which only works for scenes that
    /// are in Build Settings — so registering them had to stop being a manual
    /// chore. "Open" entries are here for the same reason: hunting for a scene
    /// under Assets/Samples/&lt;package&gt;/&lt;version&gt;/ is tedious.
    /// </summary>
    public static class LiteRtLmSampleMenu
    {
        private const string MenuRoot = "LiteRT-LM/Scenes/";

        [MenuItem(MenuRoot + "Add All Sample Scenes To Build Settings", priority = 0)]
        public static void AddAllScenesToBuildSettings()
        {
            var existing = new List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);
            var known = new HashSet<string>();
            foreach (var entry in existing)
            {
                known.Add(entry.path);
            }

            var added = 0;
            var missing = new List<string>();

            foreach (var sceneName in LiteRtLmSceneNavigator.DefaultSceneOrder)
            {
                var path = LiteRtLmSamplePaths.FindSceneByName(sceneName);
                if (string.IsNullOrEmpty(path))
                {
                    missing.Add(sceneName);
                    continue;
                }

                if (known.Add(path))
                {
                    existing.Add(new EditorBuildSettingsScene(path, true));
                    added++;
                }
            }

            EditorBuildSettings.scenes = existing.ToArray();

            var message = $"Build Settings: {added} scene(s) added, {EditorBuildSettings.scenes.Length} total.";
            if (missing.Count > 0)
            {
                message += $" Not found (sample not imported?): {string.Join(", ", missing)}.";
            }

            Debug.Log($"[LiteRT-LM] {message}");
        }

        [MenuItem(MenuRoot + "Open/Quick Start", priority = 20)]
        public static void OpenQuickStart() => OpenScene("LiteRtLmSampleScene");

        [MenuItem(MenuRoot + "Open/LLM Chat", priority = 21)]
        public static void OpenLlmChat() => OpenScene("LiteRtLmLlmChatTestScene");

        [MenuItem(MenuRoot + "Open/ASR", priority = 22)]
        public static void OpenAsr() => OpenScene("LiteRtLmAsrTestScene");

        [MenuItem(MenuRoot + "Open/Multimodal", priority = 23)]
        public static void OpenMultimodal() => OpenScene("LiteRtLmMultimodalTestScene");

        [MenuItem(MenuRoot + "Open/Voice Function Calling", priority = 24)]
        public static void OpenAsrFunctionCalling() => OpenScene("LiteRtLmAsrFunctionCallingTestScene");

        [MenuItem(MenuRoot + "Open/Multimodal Function Calling", priority = 25)]
        public static void OpenMultimodalFunctionCalling() => OpenScene("LiteRtLmMultimodalFunctionCallingTestScene");

        [MenuItem(MenuRoot + "Open/Translate", priority = 26)]
        public static void OpenTranslate() => OpenScene("LiteRtLmTranslateTestScene");

        [MenuItem(MenuRoot + "Reveal StreamingAssets Folder", priority = 60)]
        public static void RevealStreamingAssets()
        {
            var path = Application.streamingAssetsPath;
            Directory.CreateDirectory(path);
            EditorUtility.RevealInFinder(path);
        }

        private static void OpenScene(string sceneName)
        {
            var path = LiteRtLmSamplePaths.FindSceneByName(sceneName);
            if (string.IsNullOrEmpty(path))
            {
                Debug.LogWarning(
                    $"[LiteRT-LM] Scene '{sceneName}' not found. Import the sample through " +
                    "Package Manager, or run Tools/Windows/Restore-LiteRtLmSamples.ps1.");
                return;
            }

            if (EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            {
                EditorSceneManager.OpenScene(path);
            }
        }
    }
}
