using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Draws a small "◀ Prev / Next ▶" bar so the sample scenes can be walked
    /// through on a device without rebuilding.
    ///
    /// Every loaded model is released before the next scene loads
    /// (<see cref="LiteRtLmModelMemory.ReleaseAll"/>). Without that, the outgoing
    /// engine would still hold its native memory while the incoming scene starts
    /// loading its own model.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class LiteRtLmSceneNavigator : MonoBehaviour
    {
        /// <summary>Scene names in walk order. Names, not paths — they must be in Build Settings.</summary>
        [SerializeField]
        private List<string> sceneOrder = new(DefaultSceneOrder);

        [SerializeField]
        [Tooltip("Wrap from the last scene back to the first.")]
        private bool wrapAround = true;

        [SerializeField]
        [Tooltip("Width of the navigation bar in pixels.")]
        private float barWidth = 340f;

        [SerializeField]
        [Tooltip("Draw order. Lower values draw on top of the scene's own IMGUI.")]
        private int guiDepth = -100;

        private string _lastReleaseSummary = string.Empty;

        /// <summary>Scene names in walk order.</summary>
        public IReadOnlyList<string> SceneOrder => sceneOrder;

        /// <summary>
        /// Default walk order, also used to decide whether a scene should get an
        /// auto-spawned navigator.
        /// </summary>
        public static readonly string[] DefaultSceneOrder =
        {
            "LiteRtLmLlmChatTestScene",
            "LiteRtLmAsrTestScene",
            "LiteRtLmMultimodalTestScene",
            "LiteRtLmAsrFunctionCallingTestScene",
            "LiteRtLmMultimodalFunctionCallingTestScene",
            "LiteRtLmTranslateTestScene",
        };

        /// <summary>
        /// Set to false before scene load to stop the navigator from appearing.
        /// Batchmode smoke tests do not want an extra IMGUI overlay.
        /// </summary>
        public static bool AutoSpawnEnabled { get; set; } = true;

        /// <summary>
        /// Adds the bar to sample scenes that do not already contain one, so
        /// scenes authored before this component existed still get navigation
        /// without being regenerated.
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
            SceneManager.sceneLoaded += OnSceneLoaded;
            EnsureForScene(SceneManager.GetActiveScene());
        }

        private static void OnSceneLoaded(Scene scene, LoadSceneMode mode) => EnsureForScene(scene);

        private static void EnsureForScene(Scene scene)
        {
            if (!AutoSpawnEnabled || !scene.IsValid() || !scene.isLoaded)
            {
                return;
            }

            if (System.Array.IndexOf(DefaultSceneOrder, scene.name) < 0)
            {
                return;
            }

#if UNITY_2022_2_OR_NEWER
            var existing = FindObjectsByType<LiteRtLmSceneNavigator>(FindObjectsInactive.Include, FindObjectsSortMode.None);
#else
            var existing = FindObjectsOfType<LiteRtLmSceneNavigator>(true);
#endif
            foreach (var navigator in existing)
            {
                if (navigator.gameObject.scene == scene)
                {
                    return;
                }
            }

            var host = new GameObject(nameof(LiteRtLmSceneNavigator));
            SceneManager.MoveGameObjectToScene(host, scene);
            host.AddComponent<LiteRtLmSceneNavigator>();
        }

        private void OnGUI()
        {
            if (sceneOrder == null || sceneOrder.Count == 0)
            {
                return;
            }

            GUI.depth = guiDepth;

            var currentName = SceneManager.GetActiveScene().name;
            var currentIndex = sceneOrder.IndexOf(currentName);

            var width = Mathf.Min(barWidth, Screen.width - 20f);
            var area = new Rect(Screen.width - width - 10f, 10f, width, 78f);

            GUILayout.BeginArea(area, GUI.skin.box);

            var position = currentIndex >= 0
                ? $"{currentIndex + 1}/{sceneOrder.Count}"
                : "—";
            GUILayout.Label($"Sample scenes  {position}   {currentName}");

            GUILayout.BeginHorizontal();

            var previousName = Resolve(currentIndex, -1);
            GUI.enabled = previousName != null;
            if (GUILayout.Button(previousName == null ? "◀ Prev" : $"◀ {Shorten(previousName)}", GUILayout.Height(28f)))
            {
                Go(previousName);
            }

            var nextName = Resolve(currentIndex, +1);
            GUI.enabled = nextName != null;
            if (GUILayout.Button(nextName == null ? "Next ▶" : $"{Shorten(nextName)} ▶", GUILayout.Height(28f)))
            {
                Go(nextName);
            }

            GUI.enabled = true;
            GUILayout.EndHorizontal();

            if (!string.IsNullOrEmpty(_lastReleaseSummary))
            {
                GUILayout.Label(_lastReleaseSummary);
            }

            GUILayout.EndArea();
        }

        /// <summary>Next/previous scene name, or null when there is nowhere to go.</summary>
        private string Resolve(int currentIndex, int step)
        {
            if (sceneOrder.Count == 0)
            {
                return null;
            }

            // Unknown current scene: offer the first entry so the bar still works.
            if (currentIndex < 0)
            {
                return CanLoad(sceneOrder[0]) ? sceneOrder[0] : null;
            }

            // Walk past scenes that were not included in the build.
            for (var i = 1; i <= sceneOrder.Count; i++)
            {
                var index = currentIndex + (step * i);
                if (wrapAround)
                {
                    index = ((index % sceneOrder.Count) + sceneOrder.Count) % sceneOrder.Count;
                }
                else if (index < 0 || index >= sceneOrder.Count)
                {
                    return null;
                }

                if (index == currentIndex)
                {
                    return null;
                }

                var candidate = sceneOrder[index];
                if (CanLoad(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }

        private static bool CanLoad(string sceneName) =>
            !string.IsNullOrEmpty(sceneName) && Application.CanStreamedLevelBeLoaded(sceneName);

        private void Go(string sceneName)
        {
            if (!CanLoad(sceneName))
            {
                Debug.LogWarning(
                    $"[LiteRT-LM] Scene '{sceneName}' is not in Build Settings. " +
                    "Run the menu 'LiteRT-LM/Test Scenes/Generate All' or add it manually.");
                return;
            }

            var released = LiteRtLmModelMemory.ReleaseAll();
            _lastReleaseSummary = $"Released {released} model host(s) before load";
            Debug.Log($"[LiteRT-LM] Loading '{sceneName}'; released {released} model host(s) first.");

            SceneManager.LoadScene(sceneName);
        }

        /// <summary>Trims the shared prefix/suffix so the buttons stay readable on a phone.</summary>
        private static string Shorten(string sceneName)
        {
            const string prefix = "LiteRtLm";
            const string suffix = "TestScene";

            var text = sceneName;
            if (text.StartsWith(prefix, System.StringComparison.Ordinal))
            {
                text = text.Substring(prefix.Length);
            }

            if (text.EndsWith(suffix, System.StringComparison.Ordinal))
            {
                text = text.Substring(0, text.Length - suffix.Length);
            }

            return string.IsNullOrEmpty(text) ? sceneName : text;
        }
    }
}
