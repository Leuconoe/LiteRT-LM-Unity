using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmStatusHudOverlay : MonoBehaviour
    {
        [Header("LiteRT-LM Status HUD")]
        [SerializeField] private string statusFileName = "LiteRtLmAsrFunctionCallingDemo.status.txt";
        [SerializeField] private int maxLines = 8;
        [SerializeField] private float refreshIntervalSeconds = 1f;

        private string[] _tailLines = Array.Empty<string>();
        private string _resolvedStatusPath = "";
        private float _nextRefreshTime;
        private GUIStyle _boxStyle;
        private Texture2D _backgroundTexture;

        private void Update()
        {
            if (Time.realtimeSinceStartup < _nextRefreshTime)
            {
                return;
            }

            _nextRefreshTime = Time.realtimeSinceStartup + Mathf.Max(0.1f, refreshIntervalSeconds);
            RefreshTail();
        }

        private void OnDestroy()
        {
            if (_backgroundTexture != null)
            {
                Destroy(_backgroundTexture);
                _backgroundTexture = null;
            }
        }

        private void OnGUI()
        {
            EnsureBoxStyle();

            var builder = new StringBuilder();
            builder.Append("Status: ");
            builder.AppendLine(string.IsNullOrEmpty(_resolvedStatusPath) ? statusFileName : _resolvedStatusPath);
            if (_tailLines.Length == 0)
            {
                builder.Append("(no status lines yet)");
            }
            else
            {
                for (var i = 0; i < _tailLines.Length; i++)
                {
                    if (i > 0)
                    {
                        builder.AppendLine();
                    }

                    builder.Append(_tailLines[i]);
                }
            }

            var content = new GUIContent(builder.ToString());
            var width = Mathf.Min(Screen.width - 40f, 720f);
            var height = _boxStyle.CalcHeight(content, width);
            var rect = new Rect(20f, Screen.height - height - 20f, width, height);
            GUI.Label(rect, content, _boxStyle);
        }

        private void RefreshTail()
        {
            try
            {
                _resolvedStatusPath = ResolveStatusPath();
                if (string.IsNullOrEmpty(_resolvedStatusPath) || !File.Exists(_resolvedStatusPath))
                {
                    _tailLines = Array.Empty<string>();
                    return;
                }

                string text;
                using (var stream = new FileStream(_resolvedStatusPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                {
                    text = reader.ReadToEnd();
                }

                var lines = text.Replace("\r\n", "\n").Split('\n', StringSplitOptions.RemoveEmptyEntries);
                var count = Mathf.Clamp(maxLines, 1, lines.Length == 0 ? 1 : lines.Length);
                if (lines.Length <= count)
                {
                    _tailLines = lines;
                    return;
                }

                var tail = new string[count];
                Array.Copy(lines, lines.Length - count, tail, 0, count);
                _tailLines = tail;
            }
            catch (Exception ex)
            {
                _tailLines = new[] { $"(failed to read status file: {ex.Message})" };
            }
        }

        private string ResolveStatusPath()
        {
            if (string.IsNullOrWhiteSpace(statusFileName))
            {
                return string.Empty;
            }

            if (Path.IsPathRooted(statusFileName))
            {
                return statusFileName;
            }

            if (Application.isEditor)
            {
                var projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
                if (string.IsNullOrWhiteSpace(projectRoot))
                {
                    return string.Empty;
                }

                return Path.Combine(projectRoot, "Builds", "Logs", statusFileName);
            }

            return Path.Combine(Application.persistentDataPath, statusFileName);
        }

        private void EnsureBoxStyle()
        {
            if (_boxStyle != null)
            {
                return;
            }

            _backgroundTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            _backgroundTexture.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.6f));
            _backgroundTexture.Apply();

            _boxStyle = new GUIStyle(GUI.skin.label)
            {
                wordWrap = true,
                fontSize = 12,
                padding = new RectOffset(8, 8, 6, 6),
            };
            _boxStyle.normal.background = _backgroundTexture;
            _boxStyle.normal.textColor = Color.white;
        }
    }
}
