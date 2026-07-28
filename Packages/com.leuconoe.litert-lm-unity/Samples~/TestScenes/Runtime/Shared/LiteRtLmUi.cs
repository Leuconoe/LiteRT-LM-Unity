using System;
using System.Collections.Generic;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Shared IMGUI layout for the sample scenes.
    ///
    /// Every scene used to lay itself out differently — different panel widths,
    /// different places for the model selector, ad-hoc log areas. This gives them
    /// one grammar: a fixed left column for controls, a right column for output,
    /// and the same widgets for the things every scene needs (model picker,
    /// backend picker, VAD meter, transcript view, prompt input, status line).
    ///
    /// IMGUI is deliberate: these samples must run on a device with no UI prefabs,
    /// no Canvas setup and no font assets to ship.
    ///
    /// Widget conventions, so a scene added later still looks like the rest:
    /// <list type="bullet">
    /// <item>Settings chosen once per session — model, backend, language, VAD mode —
    /// use <see cref="Dropdown"/>. They collapse to one row and long file names
    /// ellipsize rather than widening the column.</item>
    /// <item>Things toggled constantly during a test — input source, engine — use
    /// <see cref="OptionRow"/>, which stays visible as a segmented control.</item>
    /// <item>File paths use <see cref="PathRow"/>, never a bare TextField: an
    /// unconstrained one widens the column to fit its text.</item>
    /// <item>Free text (prompts, tool JSON) uses <see cref="TextArea"/>; output and
    /// logs use <see cref="LogView"/> or <see cref="SelectableTextView"/>.</item>
    /// <item>Every scene ends its control column with <see cref="Status"/> so the
    /// current state is always in the same place.</item>
    /// </list>
    /// </summary>
    public static class LiteRtLmUi
    {
        public const float ControlColumnWidth = 420f;
        public const float Margin = 12f;
        public const float RowHeight = 26f;
        public const float NavigatorReservedHeight = 92f;

        private static GUIStyle s_title;
        private static GUIStyle s_sectionHeader;
        private static GUIStyle s_mono;
        private static GUIStyle s_statusOk;
        private static GUIStyle s_statusBusy;
        private static GUIStyle s_statusError;
        private static GUIStyle s_wrapLabel;
        private static Texture2D s_meterBg;
        private static Texture2D s_meterFill;
        private static Texture2D s_meterSpeech;

        public static GUIStyle Title => s_title ??= new GUIStyle(GUI.skin.label)
        {
            fontSize = 16,
            fontStyle = FontStyle.Bold,
            margin = new RectOffset(0, 0, 2, 6),
        };

        public static GUIStyle SectionHeader => s_sectionHeader ??= new GUIStyle(GUI.skin.label)
        {
            fontStyle = FontStyle.Bold,
            margin = new RectOffset(0, 0, 8, 2),
        };

        public static GUIStyle Mono => s_mono ??= new GUIStyle(GUI.skin.label)
        {
            font = GUI.skin.font,
            wordWrap = true,
            richText = false,
            alignment = TextAnchor.UpperLeft,
        };

        public static GUIStyle WrapLabel => s_wrapLabel ??= new GUIStyle(GUI.skin.label)
        {
            wordWrap = true,
        };

        /// <summary>Status line style chosen by state so scenes signal the same way.</summary>
        public static GUIStyle StatusStyle(StatusKind kind)
        {
            s_statusOk ??= Tinted(new Color(0.55f, 0.85f, 0.55f));
            s_statusBusy ??= Tinted(new Color(0.95f, 0.85f, 0.45f));
            s_statusError ??= Tinted(new Color(0.95f, 0.5f, 0.45f));

            return kind switch
            {
                StatusKind.Busy => s_statusBusy,
                StatusKind.Error => s_statusError,
                _ => s_statusOk,
            };
        }

        public enum StatusKind
        {
            Idle,
            Busy,
            Error,
        }

        private static GUIStyle Tinted(Color color)
        {
            var style = new GUIStyle(GUI.skin.label) { wordWrap = true };
            style.normal.textColor = color;
            return style;
        }

        /// <summary>
        /// Full-screen two-column frame. Returns the rects to lay controls and
        /// output into. The top-right corner is left free for the scene navigator.
        /// </summary>
        public static void BeginScreen(string title, out Rect controlRect, out Rect outputRect)
        {
            var top = Margin;
            var height = Screen.height - (Margin * 2f);
            var controlWidth = Mathf.Min(ControlColumnWidth, Screen.width * 0.42f);

            controlRect = new Rect(Margin, top, controlWidth, height);
            var outputX = Margin + controlWidth + Margin;
            outputRect = new Rect(
                outputX,
                top + NavigatorReservedHeight,
                Mathf.Max(120f, Screen.width - outputX - Margin),
                Mathf.Max(120f, height - NavigatorReservedHeight));

            _ = title;
        }

        /// <summary>Opens a bordered panel with a heading.</summary>
        public static void BeginPanel(Rect rect, string heading)
        {
            GUILayout.BeginArea(rect, GUI.skin.box);
            if (!string.IsNullOrEmpty(heading))
            {
                GUILayout.Label(heading, Title);
            }
        }

        public static void EndPanel() => GUILayout.EndArea();

        /// <summary>Section divider inside a panel.</summary>
        public static void Section(string heading) => GUILayout.Label(heading, SectionHeader);

        /// <summary>Status line with a consistent colour per state.</summary>
        public static void Status(string text, StatusKind kind = StatusKind.Idle)
        {
            GUILayout.Label(text ?? string.Empty, StatusStyle(kind));
        }

        /// <summary>
        /// Horizontal option picker used for models, backends, languages and modes.
        /// Wraps to more rows when the labels do not fit, so it never pushes the
        /// rest of the panel off-screen.
        /// </summary>
        public static int OptionRow(string label, int selectedIndex, IReadOnlyList<string> options, float availableWidth, bool enabled = true)
        {
            if (options == null || options.Count == 0)
            {
                return selectedIndex;
            }

            if (!string.IsNullOrEmpty(label))
            {
                GUILayout.Label(label, SectionHeader);
            }

            var previousEnabled = GUI.enabled;
            GUI.enabled = previousEnabled && enabled;

            var perRow = Mathf.Max(1, Mathf.FloorToInt(availableWidth / 130f));
            // Cells are width-constrained: without this a long label stretches its
            // button past the panel edge and the whole row spills out of the column.
            var cellWidth = Mathf.Max(60f, (availableWidth / perRow) - 6f);
            var result = selectedIndex;

            for (var i = 0; i < options.Count; i += perRow)
            {
                GUILayout.BeginHorizontal();
                for (var j = i; j < Mathf.Min(i + perRow, options.Count); j++)
                {
                    var isSelected = j == selectedIndex;
                    var toggled = GUILayout.Toggle(
                        isSelected,
                        new GUIContent(Ellipsize(options[j], cellWidth), options[j]),
                        GUI.skin.button,
                        GUILayout.Width(cellWidth),
                        GUILayout.Height(RowHeight));
                    if (toggled && !isSelected)
                    {
                        result = j;
                    }
                }

                GUILayout.EndHorizontal();
            }

            GUI.enabled = previousEnabled;
            return result;
        }

        /// <summary>
        /// Collapsed list picker. Long lists — audio clips, model tiers — belong
        /// here rather than in <see cref="OptionRow"/>: a wrapped button grid of
        /// twelve filenames dominates the panel and misaligns as labels vary.
        /// The list expands inline so it works on a touch device with no popups.
        /// </summary>
        public static int Dropdown(
            string label,
            int selectedIndex,
            IReadOnlyList<string> options,
            ref bool expanded,
            float availableWidth,
            bool enabled = true,
            int visibleRows = 6)
        {
            if (options == null || options.Count == 0)
            {
                return selectedIndex;
            }

            if (!string.IsNullOrEmpty(label))
            {
                GUILayout.Label(label, SectionHeader);
            }

            var previousEnabled = GUI.enabled;
            GUI.enabled = previousEnabled && enabled;

            selectedIndex = Mathf.Clamp(selectedIndex, 0, options.Count - 1);
            var current = options[selectedIndex];
            var arrow = expanded ? " ▲" : " ▼";

            if (GUILayout.Button(
                    new GUIContent(Ellipsize(current, availableWidth - 30f) + arrow, current),
                    GUILayout.Height(RowHeight)))
            {
                expanded = !expanded;
            }

            if (expanded)
            {
                var listHeight = Mathf.Min(options.Count, visibleRows) * (RowHeight + 2f);
                var scroll = GUILayout.BeginScrollView(
                    GetDropdownScroll(label ?? current),
                    GUILayout.Height(listHeight));

                for (var i = 0; i < options.Count; i++)
                {
                    var prefix = i == selectedIndex ? "✓ " : "    ";
                    if (GUILayout.Button(
                            new GUIContent(prefix + Ellipsize(options[i], availableWidth - 40f), options[i]),
                            GUI.skin.label,
                            GUILayout.Height(RowHeight)))
                    {
                        selectedIndex = i;
                        expanded = false;
                    }
                }

                GUILayout.EndScrollView();
                SetDropdownScroll(label ?? current, scroll);
            }

            GUI.enabled = previousEnabled;
            return selectedIndex;
        }

        private static readonly Dictionary<string, Vector2> s_dropdownScrolls = new();

        private static Vector2 GetDropdownScroll(string key) =>
            s_dropdownScrolls.TryGetValue(key, out var value) ? value : Vector2.zero;

        private static void SetDropdownScroll(string key, Vector2 value) => s_dropdownScrolls[key] = value;

        /// <summary>Trims a label to the given pixel width, keeping the tail readable.</summary>
        public static string Ellipsize(string text, float pixelWidth)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            // ~7 px per character for the default skin; good enough to stop overflow.
            var maxChars = Mathf.Max(6, Mathf.FloorToInt(pixelWidth / 7f));
            return text.Length <= maxChars ? text : text.Substring(0, maxChars - 1) + "…";
        }

        /// <summary>
        /// Single-line text input with a submit button. Returns true on submit
        /// (button press or Return), and clears the buffer.
        /// </summary>
        public static bool PromptRow(string controlName, ref string buffer, string submitLabel, bool enabled = true, float submitWidth = 90f)
        {
            var submitted = false;

            GUILayout.BeginHorizontal();

            var previousEnabled = GUI.enabled;
            GUI.enabled = previousEnabled && enabled;

            GUI.SetNextControlName(controlName);
            buffer = GUILayout.TextField(buffer ?? string.Empty, GUILayout.Height(RowHeight));

            var returnPressed =
                Event.current.type == EventType.KeyDown &&
                (Event.current.keyCode == KeyCode.Return || Event.current.keyCode == KeyCode.KeypadEnter) &&
                GUI.GetNameOfFocusedControl() == controlName;

            if (GUILayout.Button(submitLabel, GUILayout.Width(submitWidth), GUILayout.Height(RowHeight)) ||
                (returnPressed && enabled && !string.IsNullOrWhiteSpace(buffer)))
            {
                if (!string.IsNullOrWhiteSpace(buffer))
                {
                    submitted = true;
                    if (returnPressed)
                    {
                        Event.current.Use();
                    }
                }
            }

            GUI.enabled = previousEnabled;
            GUILayout.EndHorizontal();

            return submitted;
        }

        /// <summary>
        /// Multi-line editable text area with a fixed height, for prompts and tool
        /// definitions that the user should be able to edit in place.
        /// </summary>
        public static string TextArea(string value, float height)
        {
            return GUILayout.TextArea(value ?? string.Empty, GUILayout.Height(height));
        }

        /// <summary>
        /// Editable file path plus a Browse button.
        ///
        /// Both are width-constrained on purpose: an unconstrained TextField takes
        /// its minimum width from the text it holds, so one absolute path was
        /// enough to widen the whole control column and push Browse, Send and the
        /// mic row out of view behind a horizontal scrollbar.
        /// </summary>
        public static string PathRow(string value, float width, bool interactive, out bool browseClicked)
        {
            const float browseWidth = 72f;
            const float spacing = 6f;

            GUILayout.BeginHorizontal(GUILayout.Width(width));
            var updated = GUILayout.TextField(
                value ?? string.Empty,
                GUILayout.Width(Mathf.Max(60f, width - browseWidth - spacing)),
                GUILayout.Height(RowHeight));

            GUI.enabled = interactive;
            browseClicked = GUILayout.Button(
                "Browse", GUILayout.Width(browseWidth), GUILayout.Height(RowHeight));
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            return updated;
        }

        /// <summary>
        /// Scrolling transcript / log view. Pass the same scroll vector back each
        /// frame; set <paramref name="stickToBottom"/> to follow new lines.
        /// </summary>
        public static Vector2 LogView(IReadOnlyList<string> lines, Vector2 scroll, bool stickToBottom, float height = 0f)
        {
            var options = height > 0f
                ? new[] { GUILayout.ExpandHeight(false), GUILayout.Height(height) }
                : new[] { GUILayout.ExpandHeight(true) };

            scroll = GUILayout.BeginScrollView(scroll, options);

            if (lines == null || lines.Count == 0)
            {
                GUILayout.Label("(empty)", Mono);
            }
            else
            {
                for (var i = 0; i < lines.Count; i++)
                {
                    GUILayout.Label(lines[i], Mono);
                }
            }

            GUILayout.EndScrollView();

            if (stickToBottom && Event.current.type == EventType.Repaint)
            {
                scroll = new Vector2(scroll.x, float.MaxValue);
            }

            return scroll;
        }

        /// <summary>
        /// Selectable read-only text field. Unlike a Label the user can copy from
        /// it, which is what you want for a transcript.
        /// </summary>
        public static Vector2 SelectableTextView(string text, Vector2 scroll, float height)
        {
            scroll = GUILayout.BeginScrollView(scroll, GUILayout.Height(height));
            GUILayout.TextArea(text ?? string.Empty, Mono, GUILayout.ExpandHeight(true));
            GUILayout.EndScrollView();
            return scroll;
        }

        /// <summary>
        /// Horizontal level meter used for microphone input. <paramref name="normalized"/>
        /// is 0..1; <paramref name="speech"/> switches it to the speech colour.
        /// </summary>
        public static void LevelMeter(float normalized, bool speech, string caption = null, float height = 14f)
        {
            s_meterBg ??= SolidTexture(new Color(0.16f, 0.16f, 0.18f));
            s_meterFill ??= SolidTexture(new Color(0.35f, 0.55f, 0.85f));
            s_meterSpeech ??= SolidTexture(new Color(0.35f, 0.8f, 0.45f));

            if (!string.IsNullOrEmpty(caption))
            {
                GUILayout.Label(caption, Mono);
            }

            var rect = GUILayoutUtility.GetRect(10f, height, GUILayout.ExpandWidth(true));
            GUI.DrawTexture(rect, s_meterBg, ScaleMode.StretchToFill);

            var clamped = Mathf.Clamp01(normalized);
            if (clamped > 0f)
            {
                var fill = new Rect(rect.x, rect.y, rect.width * clamped, rect.height);
                GUI.DrawTexture(fill, speech ? s_meterSpeech : s_meterFill, ScaleMode.StretchToFill);
            }
        }

        /// <summary>
        /// Rolling VAD history strip: one column per sample, green where speech was
        /// detected. Gives the same at-a-glance picture in every scene that listens.
        /// </summary>
        public static void VadHistory(IReadOnlyList<float> levels, IReadOnlyList<bool> speech, float height = 34f)
        {
            s_meterBg ??= SolidTexture(new Color(0.16f, 0.16f, 0.18f));
            s_meterFill ??= SolidTexture(new Color(0.35f, 0.55f, 0.85f));
            s_meterSpeech ??= SolidTexture(new Color(0.35f, 0.8f, 0.45f));

            var rect = GUILayoutUtility.GetRect(10f, height, GUILayout.ExpandWidth(true));
            GUI.DrawTexture(rect, s_meterBg, ScaleMode.StretchToFill);

            if (levels == null || levels.Count == 0 || Event.current.type != EventType.Repaint)
            {
                return;
            }

            var columnWidth = Mathf.Max(1f, rect.width / levels.Count);
            for (var i = 0; i < levels.Count; i++)
            {
                var value = Mathf.Clamp01(levels[i]);
                if (value <= 0f)
                {
                    continue;
                }

                var columnHeight = rect.height * value;
                var column = new Rect(
                    rect.x + (i * columnWidth),
                    rect.yMax - columnHeight,
                    columnWidth,
                    columnHeight);

                var isSpeech = speech != null && i < speech.Count && speech[i];
                GUI.DrawTexture(column, isSpeech ? s_meterSpeech : s_meterFill, ScaleMode.StretchToFill);
            }
        }

        /// <summary>Converts a dB reading to the 0..1 range the meters expect.</summary>
        public static float NormalizeDb(float db, float floorDb = -60f, float ceilingDb = -5f)
        {
            if (float.IsNaN(db) || float.IsNegativeInfinity(db))
            {
                return 0f;
            }

            return Mathf.Clamp01((db - floorDb) / Mathf.Max(1f, ceilingDb - floorDb));
        }

        private static Texture2D SolidTexture(Color color)
        {
            var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false)
            {
                hideFlags = HideFlags.HideAndDontSave,
            };
            texture.SetPixel(0, 0, color);
            texture.Apply();
            return texture;
        }

        /// <summary>Appends to a bounded rolling log, trimming the oldest entries.</summary>
        public static void AppendLog(List<string> log, string line, int maxLines = 400)
        {
            if (log == null)
            {
                return;
            }

            log.Add(line);
            if (log.Count > maxLines)
            {
                log.RemoveRange(0, log.Count - maxLines);
            }
        }

        /// <summary>Timestamp prefix used by every scene's log so they read alike.</summary>
        public static string Stamp() => DateTime.Now.ToString("HH:mm:ss");
    }
}
