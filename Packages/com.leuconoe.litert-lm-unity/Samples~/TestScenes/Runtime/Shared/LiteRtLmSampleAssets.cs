using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Shared catalogue of the models and test assets the sample scenes offer.
    ///
    /// Scenes used to each carry their own hard-coded list, so the ASR scene and
    /// the function-calling scene disagreed on which models existed. One list also
    /// means a scene can show what is actually present: on desktop the folders are
    /// scanned, on Android the built-in list is used because StreamingAssets lives
    /// inside the APK and cannot be enumerated.
    /// </summary>
    public static class LiteRtLmSampleAssets
    {
        public sealed class ModelOption
        {
            public string Label;
            public string ModelPath;
            public string TokenizerPath;
            /// <summary>"whisper" or "qwen3" — selects the native ASR entry point.</summary>
            public string Mode = "whisper";
        }

        /// <summary>ASR tiers, ordered the way the README recommends them.</summary>
        public static readonly ModelOption[] AsrModels =
        {
            new()
            {
                Label = "base-acft-ko 5s",
                ModelPath = "ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite",
                TokenizerPath = "ASR/whisper-base-acft-ko/tokenizer.json",
            },
            new()
            {
                Label = "turbo-acft-ko 5s",
                ModelPath = "ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite",
                TokenizerPath = "ASR/whisper-turbo-acft-ko/tokenizer.json",
            },
            new()
            {
                Label = "base 30s i8",
                ModelPath = "ASR/whisper-base/whisper_base_30s_i8.tflite",
                TokenizerPath = "ASR/whisper-base/tokenizer.json",
            },
            new()
            {
                Label = "tiny 30s i8",
                ModelPath = "ASR/whisper-tiny/whisper_tiny_30s_i8.tflite",
                TokenizerPath = "ASR/whisper-tiny/tokenizer.json",
            },
            new()
            {
                Label = "qwen3-asr 0.6B",
                ModelPath = "ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite",
                TokenizerPath = "ASR/qwen3-asr-0.6b/tokenizer.json",
                Mode = "qwen3",
            },
        };

        /// <summary>LLM tiers usable as a function-calling router or chat model.</summary>
        public static readonly ModelOption[] LlmModels =
        {
            new() { Label = "qwen3-0.6b i4", ModelPath = "LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm" },
            new() { Label = "gemma3-1b i4", ModelPath = "LLM/gemma3-1b/gemma3-1b-it-int4.litertlm" },
            new() { Label = "qwen2.5-0.5b i4", ModelPath = "LLM/qwen2.5-0.5b/Qwen2.5-0.5B-Instruct_wi4b64_ekv1280.litertlm" },
            new() { Label = "lfm2.5-1.2b i4", ModelPath = "LLM/lfm2.5-1.2b/LFM2.5-1.2B-Instruct_int4.litertlm" },
            new() { Label = "gemma-4-E2B", ModelPath = "Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm" },
        };

        /// <summary>Multimodal models (image and audio input).</summary>
        public static readonly ModelOption[] MultimodalModels =
        {
            new() { Label = "gemma-4-E2B", ModelPath = "Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm" },
        };

        private static readonly string[] FallbackAudio =
        {
            "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3",
            "TestAssets/Audio/Tactical Evaluation Results Report - March 5, 2025.mp3",
            "TestAssets/Audio/현재 서울의 날씨는 흐림 입니다.mp3",
            "TestAssets/Audio/The current weather in Seoul is cloudy.mp3",
            "TestAssets/Audio/volume-볼륨 업.mp3",
            "TestAssets/Audio/volume-볼륨, 업.mp3",
            "TestAssets/Audio/volume-소리 키워줘.mp3",
            "TestAssets/Audio/volume-음량 증가.mp3",
        };

        private static readonly string[] FallbackImages =
        {
            "TestAssets/Images/apples.jpg",
            "TestAssets/Images/notebook.jpg",
            "TestAssets/Images/puppy-run.jpg",
            "TestAssets/Images/princess-snow-white.jpg",
            "TestAssets/Images/couple-together-working-seaside.jpg",
        };

        private static string[] s_audioCache;
        private static string[] s_imageCache;

        /// <summary>Audio clips available for testing, as StreamingAssets-relative paths.</summary>
        public static string[] AudioClips => s_audioCache ??=
            Scan("TestAssets/Audio", new[] { ".mp3", ".wav", ".ogg" }, FallbackAudio);

        /// <summary>Images available for testing, as StreamingAssets-relative paths.</summary>
        public static string[] Images => s_imageCache ??=
            Scan("TestAssets/Images", new[] { ".jpg", ".jpeg", ".png" }, FallbackImages);

        /// <summary>Short display label for a StreamingAssets-relative path.</summary>
        public static string Label(string relativePath) =>
            string.IsNullOrEmpty(relativePath) ? "(none)" : Path.GetFileNameWithoutExtension(relativePath);

        /// <summary>Labels for an option array, for use with the UI option row.</summary>
        public static string[] Labels(IReadOnlyList<ModelOption> options)
        {
            var labels = new string[options.Count];
            for (var i = 0; i < options.Count; i++)
            {
                labels[i] = options[i].Label;
            }

            return labels;
        }

        /// <summary>Labels for a path array.</summary>
        public static string[] Labels(IReadOnlyList<string> paths)
        {
            var labels = new string[paths.Count];
            for (var i = 0; i < paths.Count; i++)
            {
                labels[i] = Label(paths[i]);
            }

            return labels;
        }

        private static string[] Scan(string relativeFolder, string[] extensions, string[] fallback)
        {
            var root = Application.streamingAssetsPath;
            var folder = Path.Combine(root, relativeFolder);

            // Android keeps StreamingAssets inside the APK; there is nothing to enumerate.
            if (Application.platform == RuntimePlatform.Android || !Directory.Exists(folder))
            {
                return fallback;
            }

            var found = new List<string>();
            foreach (var file in Directory.GetFiles(folder))
            {
                var extension = Path.GetExtension(file).ToLowerInvariant();
                if (System.Array.IndexOf(extensions, extension) < 0)
                {
                    continue;
                }

                found.Add($"{relativeFolder}/{Path.GetFileName(file)}");
            }

            found.Sort(System.StringComparer.OrdinalIgnoreCase);
            return found.Count > 0 ? found.ToArray() : fallback;
        }
    }
}
