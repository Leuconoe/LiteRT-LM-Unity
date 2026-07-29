using System;
using System.Collections;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Result of one <see cref="ILiteRtLmTts.Speak"/> call.
    /// </summary>
    public sealed class LiteRtLmTtsResult
    {
        public bool Success;
        public string Backend;
        public string Error;
        public float Seconds;

        /// <summary>Set when the backend produced a file; empty when it played directly.</summary>
        public string WavPath = string.Empty;

        public override string ToString()
        {
            return Success
                ? $"{Backend} spoke in {Seconds:F2}s"
                : $"{Backend} failed: {Error}";
        }
    }

    /// <summary>
    /// Text to speech, driven the same way from every backend. Implementations play
    /// the audio themselves; a WAV path is reported when one exists so callers can
    /// keep the file for verification.
    ///
    /// The platform voices (<see cref="LiteRtLmSystemTts"/>) need no model files and
    /// carry no model licence, which is why they are the fallback everywhere. A
    /// neural backend implementing this interface can replace them without changing
    /// callers.
    /// </summary>
    public interface ILiteRtLmTts : IDisposable
    {
        /// <summary>Short name for logs and on-screen status, e.g. "Windows SAPI".</summary>
        string BackendName { get; }

        /// <summary>False when this platform has no voice, or none for the language.</summary>
        bool IsAvailable { get; }

        bool IsSpeaking { get; }

        /// <summary>
        /// Speaks <paramref name="text"/>, yielding until playback finishes.
        /// <paramref name="language"/> is a BCP-47-ish tag: "ko", "ko-KR", "en".
        /// </summary>
        IEnumerator Speak(string text, string language, Action<LiteRtLmTtsResult> onComplete);

        /// <summary>Stops playback immediately — the barge-in path when the mic opens.</summary>
        void Stop();
    }
}
