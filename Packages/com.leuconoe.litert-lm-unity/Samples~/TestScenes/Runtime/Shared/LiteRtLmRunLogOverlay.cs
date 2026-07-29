using System.Collections.Generic;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Status panel and rolling log for scenes that run unattended.
    ///
    /// The smoke and benchmark scenes were authored with neither a camera nor any
    /// UI, so pressing Play showed a black screen and progress was only visible in
    /// the Console. This attaches itself at runtime — no scene edits needed — and
    /// gives them the same status/log presentation the interactive scenes use.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class LiteRtLmRunLogOverlay : MonoBehaviour
    {
        /// <summary>
        /// Most recently attached overlay. The automated runners log from static
        /// helpers, so they need a way to reach the instance.
        /// </summary>
        public static LiteRtLmRunLogOverlay Current { get; private set; }

        private readonly LiteRtLmLog _log = new("Automated");
        private Vector2 _scroll;
        private string _title = "LiteRT-LM";
        private string _status = "Idle";
        private LiteRtLmUi.StatusKind _statusKind = LiteRtLmUi.StatusKind.Idle;

        /// <summary>
        /// Finds or creates the overlay for the active scene, and makes sure the
        /// scene has a camera so Play mode is not a black screen.
        /// </summary>
        public static LiteRtLmRunLogOverlay Attach(string title)
        {
            EnsureCamera();

#if UNITY_2022_2_OR_NEWER
            var existing = FindAnyObjectByType<LiteRtLmRunLogOverlay>();
#else
            var existing = FindObjectOfType<LiteRtLmRunLogOverlay>();
#endif
            if (existing == null)
            {
                var host = new GameObject(nameof(LiteRtLmRunLogOverlay));
                existing = host.AddComponent<LiteRtLmRunLogOverlay>();
            }

            existing._title = title;
            Current = existing;
            return existing;
        }

        /// <summary>
        /// Keeps the player ticking while the window is not focused.
        ///
        /// Inference runs in a coroutine that polls a Task, so without this the
        /// editor stops updating the moment focus moves elsewhere and a request
        /// appears to hang until you click back in. It is the coroutine that
        /// stalls, not the CLI process — the child process keeps running.
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        public static void KeepRunningInBackground() => Application.runInBackground = true;

        /// <summary>Adds a camera with a neutral background when the scene has none.</summary>
        public static void EnsureCamera()
        {
            Application.runInBackground = true;

            if (Camera.main != null)
            {
                return;
            }

#if UNITY_2022_2_OR_NEWER
            var anyCamera = FindAnyObjectByType<Camera>();
#else
            var anyCamera = FindObjectOfType<Camera>();
#endif
            if (anyCamera != null)
            {
                return;
            }

            var cameraObject = new GameObject("Main Camera") { tag = "MainCamera" };
            var camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.12f, 0.12f, 0.14f, 1f);
            cameraObject.transform.position = new Vector3(0f, 1f, -10f);
        }

        /// <summary>Sets the one-line status shown above the log.</summary>
        public void SetStatus(string status, LiteRtLmUi.StatusKind kind = LiteRtLmUi.StatusKind.Idle)
        {
            _status = status;
            _statusKind = kind;
        }

        /// <summary>Appends a line to the panel and the Unity console.</summary>
        public void Log(string line) => _log.Info(line);

        /// <summary>Appends a pre-formatted status line (already carries its own phase tag).</summary>
        public void LogRaw(string line) => _log.Info(line);

        public void Clear() => _log.Clear();

        private void OnEnable() => Current = this;

        private void OnGUI()
        {
            LiteRtLmUi.BeginScreen(_title, out var controlRect, out _);

            // Single wide panel: these scenes have no controls, only output.
            // Inset from the top so the shared prev/next bar stays clickable.
            var panel = new Rect(
                LiteRtLmUi.Margin,
                controlRect.y + LiteRtLmUi.NavigatorReservedHeight,
                Mathf.Max(240f, Screen.width - (LiteRtLmUi.Margin * 2f)),
                Mathf.Max(120f, controlRect.height - LiteRtLmUi.NavigatorReservedHeight));

            LiteRtLmUi.BeginPanel(panel, _title);
            LiteRtLmUi.Status(_status, _statusKind);
            GUILayout.Space(4f);
            _scroll = LiteRtLmUi.LogView(_log.Lines, _scroll, stickToBottom: true);
            LiteRtLmUi.EndPanel();
        }
    }
}
