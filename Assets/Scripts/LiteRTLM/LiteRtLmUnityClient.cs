using System;
using UnityEngine;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmUnityClient : IDisposable
    {
        private const string BridgeClassName = "com.google.ai.edge.litertlm.unity.UnityLiteRtLmBridge";

        private AndroidJavaObject _bridge;

        public bool IsAvailable
        {
            get
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                return true;
#else
                return false;
#endif
            }
        }

        public bool IsInitialized
        {
            get
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                return _bridge != null && _bridge.Call<bool>("isInitialized");
#else
                return false;
#endif
            }
        }

        public void Initialize(
            string modelPath,
            string backend = "CPU",
            string cacheDir = "",
            int maxNumTokens = 0,
            int maxNumImages = 0,
            int cpuThreads = 0,
            bool enableSpeculativeDecoding = false,
            string systemInstruction = "",
            string visionBackend = "",
            string audioBackend = "")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("modelPath is required.", nameof(modelPath));
            }

            EnsureBridge();
            var nativeLibraryDir = GetAndroidNativeLibraryDir();
            _bridge.Call(
                "initialize",
                modelPath,
                backend,
                cacheDir,
                nativeLibraryDir,
                maxNumTokens,
                maxNumImages,
                cpuThreads,
                enableSpeculativeDecoding,
                systemInstruction,
                visionBackend ?? string.Empty,
                audioBackend ?? string.Empty);
#else
            throw new PlatformNotSupportedException("LiteRT-LM Unity wrapper currently supports Android device builds only.");
#endif
        }

        public string SendMessage(string text, string extraContextJson = "")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (_bridge == null)
            {
                throw new InvalidOperationException("LiteRT-LM bridge is not initialized.");
            }

            return _bridge.Call<string>("sendMessage", text, extraContextJson);
#else
            throw new PlatformNotSupportedException("LiteRT-LM Unity wrapper currently supports Android device builds only.");
#endif
        }

        public string SendMessageWithMedia(
            string text,
            byte[] imageBytes = null,
            string imagePath = "",
            string audioPath = "",
            string extraContextJson = "")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (_bridge == null)
            {
                throw new InvalidOperationException("LiteRT-LM bridge is not initialized.");
            }

            return _bridge.Call<string>(
                "sendMessageWithMedia",
                text ?? string.Empty,
                imageBytes,
                imagePath ?? string.Empty,
                audioPath ?? string.Empty,
                extraContextJson ?? string.Empty);
#else
            throw new PlatformNotSupportedException("LiteRT-LM Unity wrapper currently supports Android device builds only.");
#endif
        }

        public string RunBenchmark(
            string modelPath,
            string backend = "CPU",
            string cacheDir = "",
            int prefillTokens = 64,
            int decodeTokens = 32,
            bool enableSpeculativeDecoding = false)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("modelPath is required.", nameof(modelPath));
            }

            EnsureBridge();
            var nativeLibraryDir = GetAndroidNativeLibraryDir();
            return _bridge.Call<string>(
                "runBenchmark",
                modelPath,
                backend,
                cacheDir,
                nativeLibraryDir,
                prefillTokens,
                decodeTokens,
                enableSpeculativeDecoding);
#else
            throw new PlatformNotSupportedException("LiteRT-LM Unity wrapper currently supports Android device builds only.");
#endif
        }

        public string InspectLiteRtModel(string modelPath)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("modelPath is required.", nameof(modelPath));
            }

            EnsureBridge();
            return _bridge.Call<string>("inspectLiteRtModel", modelPath);
#else
            throw new PlatformNotSupportedException("LiteRT model inspection currently supports Android device builds only.");
#endif
        }

        public string RunParakeetAsrSmoke(string modelPath, string audioPath, string tokenizerJsonPath, string backend = "GPU_FP16")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("modelPath is required.", nameof(modelPath));
            }
            if (string.IsNullOrWhiteSpace(audioPath))
            {
                throw new ArgumentException("audioPath is required.", nameof(audioPath));
            }
            if (string.IsNullOrWhiteSpace(tokenizerJsonPath))
            {
                throw new ArgumentException("tokenizerJsonPath is required.", nameof(tokenizerJsonPath));
            }
            if (string.IsNullOrWhiteSpace(backend))
            {
                throw new ArgumentException("backend is required.", nameof(backend));
            }

            EnsureBridge();
            return _bridge.Call<string>("runParakeetAsrSmoke", modelPath, audioPath, tokenizerJsonPath, backend);
#else
            throw new PlatformNotSupportedException("Parakeet ASR smoke test currently supports Android device builds only.");
#endif
        }

        public string RunWhisperAsrSmoke(
            string modelPath,
            string audioPath,
            string tokenizerJsonPath,
            string backend = "CPU",
            string language = "auto")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("modelPath is required.", nameof(modelPath));
            }
            if (string.IsNullOrWhiteSpace(audioPath))
            {
                throw new ArgumentException("audioPath is required.", nameof(audioPath));
            }
            if (string.IsNullOrWhiteSpace(tokenizerJsonPath))
            {
                throw new ArgumentException("tokenizerJsonPath is required.", nameof(tokenizerJsonPath));
            }
            if (string.IsNullOrWhiteSpace(backend))
            {
                throw new ArgumentException("backend is required.", nameof(backend));
            }
            if (string.IsNullOrWhiteSpace(language))
            {
                throw new ArgumentException("language is required.", nameof(language));
            }

            EnsureBridge();
            return _bridge.Call<string>("runWhisperAsrSmoke", modelPath, audioPath, tokenizerJsonPath, backend, language);
#else
            throw new PlatformNotSupportedException("Whisper ASR smoke test currently supports Android device builds only.");
#endif
        }

        public string RunQwen3AsrSmoke(
            string modelPath,
            string audioPath,
            string tokenizerJsonPath,
            string backend = "CPU",
            string language = "auto")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("modelPath is required.", nameof(modelPath));
            }
            if (string.IsNullOrWhiteSpace(audioPath))
            {
                throw new ArgumentException("audioPath is required.", nameof(audioPath));
            }
            if (string.IsNullOrWhiteSpace(tokenizerJsonPath))
            {
                throw new ArgumentException("tokenizerJsonPath is required.", nameof(tokenizerJsonPath));
            }
            if (string.IsNullOrWhiteSpace(backend))
            {
                throw new ArgumentException("backend is required.", nameof(backend));
            }
            if (string.IsNullOrWhiteSpace(language))
            {
                language = "auto";
            }

            EnsureBridge();
            return _bridge.Call<string>("runQwen3AsrSmoke", modelPath, audioPath, tokenizerJsonPath, backend, language);
#else
            throw new PlatformNotSupportedException("Qwen3 ASR smoke test currently supports Android device builds only.");
#endif
        }

        public void ResetConversation(string systemInstruction = "")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (_bridge == null)
            {
                throw new InvalidOperationException("LiteRT-LM bridge is not initialized.");
            }

            _bridge.Call("resetConversation", systemInstruction);
#endif
        }

        public void SetNativeMinLogSeverity(string severity)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            EnsureBridge();
            _bridge.Call("setNativeMinLogSeverity", severity);
#endif
        }

        public void Dispose()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (_bridge != null)
            {
                _bridge.Call("close");
                _bridge.Dispose();
                _bridge = null;
            }
#endif
            GC.SuppressFinalize(this);
        }

        ~LiteRtLmUnityClient()
        {
            Dispose();
        }

        private void EnsureBridge()
        {
            if (_bridge == null)
            {
                _bridge = new AndroidJavaObject(BridgeClassName);
            }
        }

        private static string GetAndroidNativeLibraryDir()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            using var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            using var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
            using var applicationInfo = activity.Call<AndroidJavaObject>("getApplicationInfo");
            return applicationInfo.Get<string>("nativeLibraryDir") ?? string.Empty;
#else
            return string.Empty;
#endif
        }
    }
}
