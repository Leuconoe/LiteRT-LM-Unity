using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Picks the platform voice for the current build: Android's TextToSpeech on
    /// device, SAPI on Windows. Neither needs a model file, so this is the baseline
    /// that always works and the fallback when a neural backend is unavailable.
    /// </summary>
    public static class LiteRtLmSystemTts
    {
        /// <param name="audioSource">
        /// Used by backends that play a synthesized file (Windows). Leave null to let
        /// one be created; Android speaks through the system and ignores it.
        /// </param>
        public static ILiteRtLmTts Create(AudioSource audioSource = null)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return new LiteRtLmAndroidSystemTts();
#elif UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            return new LiteRtLmWindowsSapiTts(audioSource);
#else
            return new LiteRtLmUnsupportedTts();
#endif
        }
    }

    /// <summary>Stands in on platforms with no system voice wired up yet (macOS, Linux).</summary>
    internal sealed class LiteRtLmUnsupportedTts : ILiteRtLmTts
    {
        public string BackendName => "none";

        public bool IsAvailable => false;

        public bool IsSpeaking => false;

        public System.Collections.IEnumerator Speak(
            string text, string language, System.Action<LiteRtLmTtsResult> onComplete)
        {
            onComplete?.Invoke(new LiteRtLmTtsResult
            {
                Backend = BackendName,
                Error = $"no system voice on {Application.platform}",
            });
            yield break;
        }

        public void Stop() { }

        public void Dispose() { }
    }
}
