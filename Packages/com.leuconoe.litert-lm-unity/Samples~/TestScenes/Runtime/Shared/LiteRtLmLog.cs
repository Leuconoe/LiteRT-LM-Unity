using System;
using System.Collections.Generic;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>Severity for <see cref="LiteRtLmLog"/>.</summary>
    public enum LiteRtLmLogLevel
    {
        Verbose = 0,
        Info = 1,
        Warning = 2,
        Error = 3,
        Off = 4,
    }

    /// <summary>
    /// One place that decides what gets logged and where it goes.
    ///
    /// Each scene needs two audiences at once: the on-screen panel, which is all a
    /// tester holding a device can see, and the Unity console, which is what shows
    /// up in logcat and in the editor when something fails. Scenes used to write to
    /// one or the other inconsistently — a failure would appear on screen but not
    /// in logcat, or vice versa. Routing both through here also gives a single
    /// gate: raise <see cref="MinimumLevel"/> or turn off
    /// <see cref="MirrorToConsole"/> for a quiet benchmark run.
    /// </summary>
    public sealed class LiteRtLmLog
    {
        /// <summary>Messages below this level are dropped everywhere.</summary>
        public static LiteRtLmLogLevel MinimumLevel { get; set; } = LiteRtLmLogLevel.Info;

        /// <summary>When false, messages go to the on-screen panel only.</summary>
        public static bool MirrorToConsole { get; set; } = true;

        /// <summary>Lines kept in memory per log; older lines are dropped.</summary>
        public static int MaxLines { get; set; } = 400;

        private readonly List<string> _lines = new();
        private readonly object _gate = new();
        private string[] _snapshot = System.Array.Empty<string>();
        private readonly string _tag;

        public LiteRtLmLog(string tag) => _tag = string.IsNullOrEmpty(tag) ? "LiteRT-LM" : tag;

        /// <summary>
        /// Buffered lines, oldest first. A snapshot rather than the live list:
        /// benchmarks log from a worker thread while OnGUI iterates on the main
        /// thread, and handing out the mutable list throws mid-frame.
        /// </summary>
        public IReadOnlyList<string> Lines
        {
            get
            {
                lock (_gate)
                {
                    return _snapshot;
                }
            }
        }

        /// <summary>Number of buffered lines.</summary>
        public int Count
        {
            get
            {
                lock (_gate)
                {
                    return _lines.Count;
                }
            }
        }

        /// <summary>Detail that is useful while debugging but noise otherwise.</summary>
        public void Verbose(string message) => Write(LiteRtLmLogLevel.Verbose, message);

        /// <summary>Normal progress: what was selected, what was sent, how long it took.</summary>
        public void Info(string message) => Write(LiteRtLmLogLevel.Info, message);

        /// <summary>Something recoverable: a missing optional asset, a fallback path.</summary>
        public void Warning(string message) => Write(LiteRtLmLogLevel.Warning, message);

        /// <summary>The operation failed.</summary>
        public void Error(string message) => Write(LiteRtLmLogLevel.Error, message);

        /// <summary>Logs an exception at error level, keeping the stack trace in the console.</summary>
        public void Exception(string context, Exception exception)
        {
            Write(LiteRtLmLogLevel.Error, $"{context}: {exception?.Message}");
            if (MirrorToConsole && exception != null)
            {
                Debug.LogException(exception);
            }
        }

        public void Clear()
        {
            lock (_gate)
            {
                _lines.Clear();
                _snapshot = System.Array.Empty<string>();
            }
        }

        /// <summary>Whole buffer as text, for copying to the clipboard.</summary>
        public string ToPlainText() => string.Join(Environment.NewLine, Lines);

        private void Write(LiteRtLmLogLevel level, string message)
        {
            if (level < MinimumLevel || MinimumLevel == LiteRtLmLogLevel.Off)
            {
                return;
            }

            var stamped = $"[{LiteRtLmUi.Stamp()}] {message}";
            lock (_gate)
            {
                _lines.Add(stamped);
                if (_lines.Count > MaxLines)
                {
                    _lines.RemoveRange(0, _lines.Count - MaxLines);
                }

                _snapshot = _lines.ToArray();
            }

            if (!MirrorToConsole)
            {
                return;
            }

            var line = $"[LiteRT-LM {_tag}] {message}";
            switch (level)
            {
                case LiteRtLmLogLevel.Error:
                    Debug.LogError(line);
                    break;
                case LiteRtLmLogLevel.Warning:
                    Debug.LogWarning(line);
                    break;
                default:
                    Debug.Log(line);
                    break;
            }
        }
    }
}
