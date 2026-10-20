using System;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// SenseVoice-Small ASR on the fixed-shape LiteRT buckets of litert-community/SenseVoiceSmall (or
    /// …-ko-zeroth), through <c>UnityLiteRtLmBridge.runSenseVoiceAsr</c> (AAR take20+). The front end
    /// (fbank/LFR/CMVN), bucket choice and greedy CTC run natively; the model folder holds am.mvn,
    /// tokens.txt and <c>sensevoice_small_{3s,5s,10s,30s}_{variant}.tflite</c> (only the buckets you ship
    /// are needed — a clip uses the smallest present one that fits, longer audio is truncated to the largest
    /// present; each bucket is a full ~242 MB copy of the weights). Android device builds only.
    /// </summary>
    public sealed class LiteRtLmSenseVoice
    {
        private const string BridgeClassName = "com.google.ai.edge.litertlm.unity.UnityLiteRtLmBridge";

        /// <summary>
        /// Suggested threshold on <see cref="Confidence.meanMaxProb"/> below which an app should ask the user to
        /// confirm what was heard. It flags transcription errors (AUROC ≈ 0.73 on synthetic Korean commands) but
        /// cannot catch a correct transcript that the downstream logic misreads; 0.93 is the stricter option.
        /// </summary>
        public const double RecommendedMinConfidence = 0.90;

        /// <summary>CTC confidence of one transcription — what to gate a voice command on.</summary>
        [Serializable]
        public sealed class Confidence
        {
            /// <summary>Mean over the speech frames of the best softmax probability.</summary>
            public float meanMaxProb;

            /// <summary>Lowest per-token probability (0 when nothing was recognised).</summary>
            public float minTokenProb;

            /// <summary>Each emitted token's best probability over its CTC run (tags excluded).</summary>
            public float[] perToken;
        }

        [Serializable]
        public sealed class Result
        {
            public bool success;
            public string text;
            public int[] tokens;
            public Confidence confidence;
            public string bucket;
            public int frames;
            public float audioSeconds, featMs, inferMs, decodeMs, compileMs;
            public string error;

            /// <summary>The native JSON, kept for logging.</summary>
            [NonSerialized] public string Json;
        }

        private readonly string _modelDir;
        private readonly string _variant;

        /// <param name="variant"><c>w8</c> (weights int8, fp32 accuracy) or <c>i8</c> (~25 % faster, worse in noise).</param>
        public LiteRtLmSenseVoice(string modelDir, string variant = "w8")
        {
            _modelDir = modelDir ?? throw new ArgumentNullException(nameof(modelDir));
            _variant = variant;
        }

        /// <summary>
        /// Transcribes any audio container miniaudio decodes (wav/mp3/flac). <paramref name="language"/>: ko (default),
        /// en, zh, ja, yue, auto. <paramref name="dumpDir"/> non-empty writes feats.f32 / logits.f32 for parity checks.
        /// </summary>
        public Result Transcribe(string audioPath, string language = "ko", bool useItn = false, string dumpDir = "")
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            using var bridge = new AndroidJavaClass(BridgeClassName);
            var json = bridge.CallStatic<string>("runSenseVoiceAsr", _modelDir, _variant, audioPath, language, useItn, dumpDir ?? "");
            var result = JsonUtility.FromJson<Result>(json) ?? new Result { error = "empty response" };
            result.Json = json;
            return result;
#else
            throw new PlatformNotSupportedException("SenseVoice runs through the Android bridge only.");
#endif
        }
    }
}
