using System;
using System.Collections;
using System.IO;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Speech-to-text on desktop.
    ///
    /// The whisper tflite path lives in the Android JNI bridge, so in the editor
    /// the ASR scenes could only save a wav and stop. `litert_lm_advanced_main`
    /// accepts audio as an `[audio:...]` prompt tag, so a multimodal model can
    /// transcribe instead — the same trick the translate scene uses. It is not the
    /// same engine as on device (different model, different latency), and the UI
    /// says so, but it makes the scenes testable without an APK.
    /// </summary>
    public static class LiteRtLmDesktopAsr
    {
        /// <summary>Model used for desktop transcription, relative to StreamingAssets.</summary>
        public const string DefaultModelRelativePath = "Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm";

        /// <summary>Advanced CLI, relative to the project root.</summary>
        public const string DefaultExecutableRelativePath =
            "Tools/Windows/Bin/litert_lm_advanced_main.windows_x86_64.exe";

        /// <summary>True when this platform can run the desktop ASR path.</summary>
        public static bool IsSupported =>
            Application.platform == RuntimePlatform.WindowsEditor ||
            Application.platform == RuntimePlatform.WindowsPlayer;

        /// <summary>Result of one desktop transcription.</summary>
        public sealed class Result
        {
            public bool Success;
            public string Transcript = string.Empty;
            public string Error;
            public float ElapsedSeconds;
            /// <summary>Raw model output, kept for the log when parsing looks off.</summary>
            public string Raw = string.Empty;
        }

        /// <summary>
        /// Transcribes a wav/mp3 file. Yields until the CLI finishes; the result is
        /// handed to <paramref name="onComplete"/>.
        /// </summary>
        /// <param name="audioPath">Absolute path to the audio file.</param>
        /// <param name="language">"ko", "en" or "auto" — steers the instruction only.</param>
        /// <param name="backend">"CPU" or "GPU".</param>
        public static IEnumerator Transcribe(
            string audioPath,
            string language,
            string backend,
            Action<Result> onComplete)
        {
            var result = new Result();

            if (!IsSupported)
            {
                result.Error = "Desktop ASR is Windows-only.";
                onComplete(result);
                yield break;
            }

            if (string.IsNullOrEmpty(audioPath) || !File.Exists(audioPath))
            {
                result.Error = $"Audio file not found: {audioPath}";
                onComplete(result);
                yield break;
            }

            var modelPath = LiteRtLmStreamingAssets.Resolve(DefaultModelRelativePath);
            if (modelPath == null)
            {
                result.Error =
                    $"Desktop ASR needs {DefaultModelRelativePath} in StreamingAssets " +
                    $"(available: {LiteRtLmStreamingAssets.DescribeAvailable()}).";
                onComplete(result);
                yield break;
            }

            var executable = Path.Combine(ProjectRoot(), DefaultExecutableRelativePath);
            if (!File.Exists(executable))
            {
                result.Error = $"Advanced CLI not found: {executable}";
                onComplete(result);
                yield break;
            }

            // Two constraints on the [audio:...] tag, both discovered the hard way:
            //   1. it decodes wav only — an mp3 path is passed through as text;
            //   2. the parser stops at the first space, so a path containing spaces
            //      is truncated and the tail leaks into the prompt. That is why a
            //      clip called "volume-소리 키워줘.wav" came back transcribed as its
            //      own filename.
            // EnsureWav writes a space-free ASCII name, so both are handled here.
            if (!audioPath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase) ||
                NeedsSafeName(audioPath))
            {
                string converted = null;
                string convertError = null;
                yield return EnsureWav(audioPath, p => converted = p, e => convertError = e);

                if (converted == null)
                {
                    result.Error = $"Could not convert '{Path.GetFileName(audioPath)}' to wav: {convertError}";
                    onComplete(result);
                    yield break;
                }

                audioPath = converted;
            }

            var languageHint = language switch
            {
                "ko" => " The speech is Korean; reply in Korean.",
                "en" => " The speech is English; reply in English.",
                _ => string.Empty,
            };

            // Instruction first, tag last — this is the order upstream documents
            // ("Transcribe the audio [audio:/path/to/audio.wav]"). With the tag
            // leading, short clips made the model continue the prompt text instead
            // of answering, so the instruction leaked into the transcript.
            var instruction =
                "Transcribe the speech in this audio exactly. Output only the transcription." + languageHint;
            var prompt = instruction + " [audio:" + audioPath.Replace("\\", "/") + "]";

            var startedAt = Time.realtimeSinceStartup;
            var client = new LiteRtLmWindowsCliClient();
            var task = System.Threading.Tasks.Task.Run(() =>
                client.SendMessage(executable, modelPath, prompt, string.IsNullOrEmpty(backend) ? "CPU" : backend));

            while (!task.IsCompleted)
            {
                yield return null;
            }

            result.ElapsedSeconds = Time.realtimeSinceStartup - startedAt;

            if (task.IsFaulted)
            {
                result.Error = task.Exception?.GetBaseException().Message ?? "unknown CLI failure";
                onComplete(result);
                yield break;
            }

            result.Raw = task.Result ?? string.Empty;
            result.Transcript = Clean(StripEcho(result.Raw, instruction));
            result.Success = !string.IsNullOrWhiteSpace(result.Transcript);
            if (!result.Success)
            {
                result.Error = "The model returned an empty transcription.";
            }

            onComplete(result);
        }

        /// <summary>
        /// Returns a 16 kHz mono wav for any format Unity can decode (mp3, ogg, …),
        /// cached between runs. The CLI's [audio:] tag only reads wav, so every
        /// desktop audio path has to go through here first.
        /// </summary>
        public static IEnumerator EnsureWav(string sourcePath, Action<string> onDone, Action<string> onError)
        {
            var cacheDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "WavCache");
            Directory.CreateDirectory(cacheDirectory);
            var target = Path.Combine(cacheDirectory, SafeName(sourcePath) + ".wav");

            if (File.Exists(target))
            {
                onDone(target);
                yield break;
            }

            // Already a wav: the only problem is the name, so copy rather than decode.
            if (sourcePath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
            {
                try
                {
                    File.Copy(sourcePath, target, true);
                    onDone(target);
                }
                catch (Exception ex)
                {
                    onError(ex.Message);
                }

                yield break;
            }

            var audioType = Path.GetExtension(sourcePath).ToLowerInvariant() switch
            {
                ".mp3" => AudioType.MPEG,
                ".ogg" => AudioType.OGGVORBIS,
                ".wav" => AudioType.WAV,
                _ => AudioType.UNKNOWN,
            };

            if (audioType == AudioType.UNKNOWN)
            {
                onError($"unsupported format '{Path.GetExtension(sourcePath)}'");
                yield break;
            }

            var uri = new Uri(sourcePath).AbsoluteUri;
            using var request = UnityEngine.Networking.UnityWebRequestMultimedia.GetAudioClip(uri, audioType);
            yield return request.SendWebRequest();

            if (request.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                onError(request.error);
                yield break;
            }

            var clip = UnityEngine.Networking.DownloadHandlerAudioClip.GetContent(request);
            if (clip == null || clip.samples == 0)
            {
                onError("decoded clip is empty");
                yield break;
            }

            var samples = new float[clip.samples * clip.channels];
            clip.GetData(samples, 0);

            // Downmix to mono, then resample to the 16 kHz the models expect.
            var mono = new float[clip.samples];
            for (var i = 0; i < clip.samples; i++)
            {
                var sum = 0f;
                for (var c = 0; c < clip.channels; c++)
                {
                    sum += samples[(i * clip.channels) + c];
                }

                mono[i] = sum / clip.channels;
            }

            var resampled = Resample(mono, clip.frequency, LiteRtLmMicVadCapture.TargetSampleRate);
            LiteRtLmMicVadCapture.WriteWav16BitMono(target, resampled, LiteRtLmMicVadCapture.TargetSampleRate);
            onDone(target);
        }

        /// <summary>Linear resample; adequate for speech recognition input.</summary>
        private static float[] Resample(float[] input, int sourceRate, int targetRate)
        {
            if (sourceRate == targetRate || input.Length == 0)
            {
                return input;
            }

            var ratio = targetRate / (double)sourceRate;
            var length = Math.Max(1, (int)(input.Length * ratio));
            var output = new float[length];

            for (var i = 0; i < length; i++)
            {
                var position = i / ratio;
                var index = (int)position;
                var fraction = (float)(position - index);
                var a = input[Math.Min(index, input.Length - 1)];
                var b = input[Math.Min(index + 1, input.Length - 1)];
                output[i] = Mathf.Lerp(a, b, fraction);
            }

            return output;
        }

        /// <summary>
        /// Removes the instruction when the model echoes it back. Short clips make
        /// this likely: with little speech to transcribe the model continues the
        /// prompt instead, and the instruction ends up glued to the transcript.
        /// </summary>
        private static string StripEcho(string raw, string instruction)
        {
            if (string.IsNullOrEmpty(raw) || string.IsNullOrEmpty(instruction))
            {
                return raw;
            }

            var text = raw;
            var index = text.IndexOf(instruction, StringComparison.OrdinalIgnoreCase);
            if (index >= 0)
            {
                text = text.Remove(index, instruction.Length);
            }

            // Also drop a trailing fragment of the instruction ("...no quotes, labels").
            foreach (var fragment in new[]
                     {
                         "Transcribe the speech in this audio exactly",
                         "Output only the transcription",
                         "The speech is Korean; reply in Korean",
                         "The speech is English; reply in English",
                     })
            {
                var at = text.IndexOf(fragment, StringComparison.OrdinalIgnoreCase);
                if (at >= 0)
                {
                    text = text.Substring(0, at);
                }
            }

            return text.Trim();
        }

        /// <summary>
        /// Strips the wrapping a chat model tends to add — surrounding quotes and a
        /// leading "Transcription:"-style label.
        /// </summary>
        public static string Clean(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
            {
                return string.Empty;
            }

            var text = raw.Trim();

            var colon = text.IndexOf(':');
            if (colon > 0 && colon < 24)
            {
                var head = text.Substring(0, colon).ToLowerInvariant();
                if (head.Contains("transcription") || head.Contains("transcript") || head.Contains("audio"))
                {
                    text = text.Substring(colon + 1).Trim();
                }
            }

            if (text.Length >= 2)
            {
                var first = text[0];
                var last = text[text.Length - 1];
                if ((first == '"' && last == '"') || (first == '\'' && last == '\'') ||
                    (first == '“' && last == '”'))
                {
                    text = text.Substring(1, text.Length - 2).Trim();
                }
            }

            return text;
        }

        /// <summary>
        /// Returns a path safe to place inside an [image:]/[audio:] tag, copying the
        /// file into the cache under an ASCII, space-free name when necessary. The
        /// tag parser stops at the first space, so any path containing one is
        /// silently truncated.
        /// </summary>
        public static string SafeMediaPath(string sourcePath)
        {
            if (string.IsNullOrEmpty(sourcePath) || !NeedsSafeName(sourcePath))
            {
                return sourcePath;
            }

            try
            {
                var cacheDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "MediaCache");
                Directory.CreateDirectory(cacheDirectory);
                var target = Path.Combine(
                    cacheDirectory,
                    SafeName(sourcePath) + Path.GetExtension(sourcePath));

                if (!File.Exists(target))
                {
                    File.Copy(sourcePath, target, true);
                }

                return target;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[LiteRT-LM] Could not stage '{sourcePath}' under a safe name: {ex.Message}");
                return sourcePath;
            }
        }

        /// <summary>True when a path would break the whitespace-delimited [audio:] tag.</summary>
        private static bool NeedsSafeName(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                return false;
            }

            foreach (var c in path)
            {
                if (c == ' ' || c > 127)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// ASCII, space-free cache name derived from the source path. The hash keeps
        /// distinct sources apart when their sanitized names would collide.
        /// </summary>
        private static string SafeName(string sourcePath)
        {
            var stem = Path.GetFileNameWithoutExtension(sourcePath) ?? "audio";
            var builder = new System.Text.StringBuilder(stem.Length);
            foreach (var c in stem)
            {
                builder.Append(char.IsLetterOrDigit(c) && c < 128 ? c : '_');
            }

            var ascii = builder.ToString().Trim('_');
            if (ascii.Length == 0)
            {
                ascii = "audio";
            }

            var hash = (uint)sourcePath.GetHashCode();
            return $"{ascii}_{hash:x8}";
        }

        private static string ProjectRoot() =>
            Path.GetDirectoryName(Application.dataPath) ?? Directory.GetCurrentDirectory();
    }
}
