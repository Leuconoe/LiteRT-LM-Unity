using System;
using System.Collections;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Android's built-in <c>android.speech.tts.TextToSpeech</c>, reached through
    /// <see cref="AndroidJavaObject"/>. No AAR change, no native build, no model
    /// files — it uses whatever engine and voice data the device already has.
    ///
    /// Korean depends on the device: <c>isLanguageAvailable</c> is checked and
    /// reported honestly rather than failing silently at speak time.
    /// </summary>
    public sealed class LiteRtLmAndroidSystemTts : ILiteRtLmTts
    {
        // TextToSpeech.SUCCESS / QUEUE_FLUSH / LANG_MISSING_DATA / LANG_NOT_SUPPORTED
        private const int InitSuccess = 0;
        private const int QueueFlush = 0;
        private const int LangMissingData = -1;
        private const int LangNotSupported = -2;

        private const float InitTimeoutSeconds = 8f;
        private const float SpeakTimeoutSeconds = 60f;

        private AndroidJavaObject _tts;
#if UNITY_ANDROID && !UNITY_EDITOR
        private InitListener _listener;
#endif
        private bool _initFailed;
        private string _initError = string.Empty;
        private string _appliedLanguage = string.Empty;

        public string BackendName => "Android system TTS";

        public bool IsAvailable
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            get { return !_initFailed; }
#else
            get { return false; }
#endif
        }

        public bool IsSpeaking
        {
            get
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                try { return _tts != null && _tts.Call<bool>("isSpeaking"); }
                catch (Exception) { return false; }
#else
                return false;
#endif
            }
        }

        /// <summary>Engine init status, once known. Empty while still starting.</summary>
        public string LastError => _initError;

        public IEnumerator Speak(string text, string language, Action<LiteRtLmTtsResult> onComplete)
        {
            var result = new LiteRtLmTtsResult { Backend = BackendName };
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(text))
            {
                result.Error = "empty text";
                onComplete?.Invoke(result);
                yield break;
            }

            var startedAt = Time.realtimeSinceStartup;

            foreach (var step in EnsureEngine())
            {
                yield return step;
            }

            if (_tts == null)
            {
                result.Error = string.IsNullOrEmpty(_initError) ? "TextToSpeech unavailable" : _initError;
                onComplete?.Invoke(result);
                yield break;
            }

            var languageError = ApplyLanguage(language);
            if (languageError != null)
            {
                result.Error = languageError;
                onComplete?.Invoke(result);
                yield break;
            }

            int queued;
            try
            {
                queued = _tts.Call<int>(
                    "speak", text, QueueFlush, (AndroidJavaObject)null, "litertlm-tts");
            }
            catch (Exception exception)
            {
                result.Error = exception.Message;
                onComplete?.Invoke(result);
                yield break;
            }

            if (queued != InitSuccess)
            {
                result.Error = $"speak() returned {queued}";
                onComplete?.Invoke(result);
                yield break;
            }

            // UtteranceProgressListener is an abstract class, which AndroidJavaProxy
            // cannot implement, so completion is polled instead. Give the engine a
            // moment to report speaking before treating silence as "done".
            var speakingStarted = false;
            while (Time.realtimeSinceStartup - startedAt < SpeakTimeoutSeconds)
            {
                var speaking = IsSpeaking;
                if (speaking)
                {
                    speakingStarted = true;
                }
                else if (speakingStarted || Time.realtimeSinceStartup - startedAt > 1.5f)
                {
                    break;
                }

                yield return null;
            }

            result.Success = true;
            result.Seconds = Time.realtimeSinceStartup - startedAt;
            onComplete?.Invoke(result);
#else
            result.Error = "Android system TTS is only available on an Android device build.";
            onComplete?.Invoke(result);
            yield break;
#endif
        }

        public void Stop()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try { _tts?.Call<int>("stop"); }
            catch (Exception) { /* stopping a dead engine is not an error worth surfacing */ }
#endif
        }

        public void Dispose()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                _tts?.Call("shutdown");
                _tts?.Dispose();
            }
            catch (Exception) { }
            _listener = null;
#endif
            _tts = null;
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private IEnumerable EnsureEngine()
        {
            if (_tts != null || _initFailed)
            {
                yield break;
            }

            using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (var activity = player.GetStatic<AndroidJavaObject>("currentActivity"))
            {
                _listener = new InitListener();
                _tts = new AndroidJavaObject("android.speech.tts.TextToSpeech", activity, _listener);
            }

            var deadline = Time.realtimeSinceStartup + InitTimeoutSeconds;
            while (!_listener.Completed && Time.realtimeSinceStartup < deadline)
            {
                yield return null;
            }

            if (!_listener.Completed)
            {
                _initError = "TextToSpeech init timed out";
            }
            else if (_listener.Status != InitSuccess)
            {
                _initError = $"TextToSpeech init failed with status {_listener.Status}";
            }

            if (!string.IsNullOrEmpty(_initError))
            {
                _initFailed = true;
                Dispose();
            }
        }

        /// <summary>Returns null on success, otherwise the reason the language is unusable.</summary>
        private string ApplyLanguage(string language)
        {
            var tag = NormalizeLanguage(language);
            if (_appliedLanguage == tag)
            {
                return null;
            }

            var parts = tag.Split('-');
            try
            {
                using (var locale = parts.Length > 1
                    ? new AndroidJavaObject("java.util.Locale", parts[0], parts[1])
                    : new AndroidJavaObject("java.util.Locale", parts[0]))
                {
                    var available = _tts.Call<int>("isLanguageAvailable", locale);
                    if (available == LangMissingData)
                    {
                        return $"{tag}: voice data is not installed on this device";
                    }

                    if (available == LangNotSupported)
                    {
                        return $"{tag}: not supported by the device TTS engine";
                    }

                    var applied = _tts.Call<int>("setLanguage", locale);
                    if (applied == LangMissingData || applied == LangNotSupported)
                    {
                        return $"{tag}: setLanguage returned {applied}";
                    }
                }
            }
            catch (Exception exception)
            {
                return exception.Message;
            }

            _appliedLanguage = tag;
            return null;
        }

        private sealed class InitListener : AndroidJavaProxy
        {
            public InitListener() : base("android.speech.tts.TextToSpeech$OnInitListener") { }

            public bool Completed { get; private set; }
            public int Status { get; private set; } = -1;

            // Called by Android on the main thread.
            public void onInit(int status)
            {
                Status = status;
                Completed = true;
            }
        }
#endif

        internal static string NormalizeLanguage(string language)
        {
            if (string.IsNullOrWhiteSpace(language))
            {
                return "ko-KR";
            }

            var tag = language.Trim().Replace('_', '-');
            switch (tag.ToLowerInvariant())
            {
                case "ko": return "ko-KR";
                case "en": return "en-US";
                default: return tag;
            }
        }
    }
}
