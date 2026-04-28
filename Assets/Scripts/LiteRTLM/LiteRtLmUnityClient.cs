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
            string systemInstruction = "")
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
                systemInstruction);
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

        public string RunBenchmark(
            string modelPath,
            string backend = "CPU",
            string cacheDir = "",
            int prefillTokens = 64,
            int decodeTokens = 32)
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
                decodeTokens);
#else
            throw new PlatformNotSupportedException("LiteRT-LM Unity wrapper currently supports Android device builds only.");
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
