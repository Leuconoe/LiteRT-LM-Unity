using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmAsrTestRunner : MonoBehaviour
    {
        private const string QwenAsrMode = "qwen3-asr";

        private static readonly string[] AudioOptions =
        {
            "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3",
            "TestAssets/Audio/Tactical Evaluation Results Report - March 5, 2025.mp3",
            "TestAssets/Audio/현재 서울의 날씨는, 흐림. 입니다.mp3",
            "TestAssets/Audio/The current weather in Seoul is cloudy.mp3",
            "TestAssets/Audio/변경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다..mp3",
            "TestAssets/Audio/We are currently reviewing the changes Additional issues. may be discovered during the app review process..mp3",
            "TestAssets/Audio/volume-볼륨 업.mp3",
            "TestAssets/Audio/volume-볼륨, 업.mp3",
            "TestAssets/Audio/volume-소리 키워줘.mp3",
            "TestAssets/Audio/volume-음량 증가.mp3",
        };

        // Reference transcripts per audio option, used for a display-only
        // expected-match indicator (never fed to the model).
        private static readonly string[] ExpectedTranscripts =
        {
            "2025년 3월 5일 전술평가 결과 보고",
            "Tactical Evaluation Results Report - March 5, 2025",
            "현재 서울의 날씨는, 흐림. 입니다",
            "The current weather in Seoul is cloudy",
            "변경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다.",
            "We are currently reviewing the changes Additional issues. may be discovered during the app review process.",
            "볼륨 업",
            "볼륨 업",
            "소리 키워줘",
            "음량 증가",
        };

        [Serializable]
        public sealed class AsrModelOption
        {
            public string label;
            public string modelPath;
            public string tokenizerJsonPath;
            public string mode;
        }

        [Header("LiteRT-LM ASR Test")]
        [SerializeField] private AsrModelOption[] modelOptions =
        {
            new AsrModelOption
            {
                label = "Whisper Tiny i8",
                modelPath = "ASR/whisper-tiny/whisper_tiny_30s_i8.tflite",
                tokenizerJsonPath = "ASR/whisper-tiny/tokenizer.json",
                mode = "whisper",
            },
            new AsrModelOption
            {
                label = "Whisper Base f32",
                modelPath = "ASR/whisper-base/whisper_base_30s_f32.tflite",
                tokenizerJsonPath = "ASR/whisper-base/tokenizer.json",
                mode = "whisper",
            },
            new AsrModelOption
            {
                label = "Qwen3-ASR-0.6B i8",
                modelPath = "ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite",
                tokenizerJsonPath = "ASR/qwen3-asr-0.6b/tokenizer.json",
                mode = QwenAsrMode,
            },
            new AsrModelOption
            {
                label = "Whisper Base ACFT-KO 5s (voice commands)",
                modelPath = "ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite",
                tokenizerJsonPath = "ASR/whisper-base-acft-ko/tokenizer.json",
                mode = "whisper",
            },
            new AsrModelOption
            {
                label = "Whisper Medium ACFT-KO 5s",
                modelPath = "ASR/whisper-medium-acft-ko/acft_medium_5s_drq.tflite",
                tokenizerJsonPath = "ASR/whisper-medium-acft-ko/tokenizer.json",
                mode = "whisper",
            },
            new AsrModelOption
            {
                label = "Whisper Turbo ACFT-KO 5s",
                modelPath = "ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite",
                tokenizerJsonPath = "ASR/whisper-turbo-acft-ko/tokenizer.json",
                mode = "whisper",
            },
        };

        [SerializeField] private int selectedModelIndex;
        [SerializeField] private int selectedAudioIndex;
        [SerializeField] private string language = "ko";
        [SerializeField] private string backend = "CPU";

        private LiteRtLmUnityClient _client;
        private readonly List<string> _transcriptLog = new List<string>();
        private string _status = "Idle";
        private bool _isBusy;
        private bool _modelListExpanded;
        private bool _audioListExpanded;
        private float _requestStartedAt;
        private Vector2 _logScroll;
        private bool _hasImeTextFieldFocus;

        private void Awake()
        {
            _client ??= new LiteRtLmUnityClient();
        }

        private void OnDestroy()
        {
            _client?.Dispose();
            _client = null;
        }

        private void OnGUI()
        {
            _client ??= new LiteRtLmUnityClient();

            GUILayout.BeginArea(new Rect(20, 20, 760, 860), GUI.skin.box);
            GUILayout.Label("LiteRT-LM ASR Test");
            GUILayout.Label($"Status: {_status}");
            if (_isBusy)
            {
                GUILayout.Label($"Elapsed: {Time.realtimeSinceStartup - _requestStartedAt:0.0}s");
            }

            if (!_client.IsAvailable)
            {
                GUILayout.Label("ASR requires Android build (Windows CLI ASR pending)");
            }

            _hasImeTextFieldFocus = false;
            DrawModelDropdown();
            DrawAudioDropdown();

            GUILayout.Label("Language");
            language = DrawImeTextField(language, "LiteRtLmAsrLanguageField");
            GUILayout.Label("Backend");
            backend = DrawImeTextField(backend, "LiteRtLmAsrBackendField");
            if (!_hasImeTextFieldFocus)
            {
                Input.imeCompositionMode = IMECompositionMode.Auto;
            }

            var selectedOption = GetSelectedModelOption();
            var transcribeEnabled = _client.IsAvailable && !_isBusy && selectedOption != null && IsOptionEnabled(selectedOption);
            GUI.enabled = transcribeEnabled;
            if (GUILayout.Button("Transcribe", GUILayout.Height(40)))
            {
                StartCoroutine(TranscribeRoutine(selectedOption));
            }

            GUI.enabled = true;

            GUILayout.Space(8);
            GUILayout.Label("Transcript / Metrics");
            _logScroll = GUILayout.BeginScrollView(_logScroll, GUILayout.Height(440));
            foreach (var entry in _transcriptLog)
            {
                GUILayout.TextArea(entry);
                GUILayout.Space(4);
            }

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private void DrawModelDropdown()
        {
            GUILayout.Label("ASR Model");
            if (modelOptions == null || modelOptions.Length == 0)
            {
                GUILayout.Label("No ASR model options configured.");
                return;
            }

            selectedModelIndex = Mathf.Clamp(selectedModelIndex, 0, modelOptions.Length - 1);
            if (GUILayout.Button($"{GetOptionLabel(modelOptions[selectedModelIndex])} {(_modelListExpanded ? "▲" : "▼")}"))
            {
                _modelListExpanded = !_modelListExpanded;
            }

            if (!_modelListExpanded)
            {
                return;
            }

            for (var i = 0; i < modelOptions.Length; i++)
            {
                var option = modelOptions[i];
                GUI.enabled = IsOptionEnabled(option);
                if (GUILayout.Button((i == selectedModelIndex ? "✓ " : "   ") + GetOptionLabel(option)))
                {
                    selectedModelIndex = i;
                    _modelListExpanded = false;
                }

                GUI.enabled = true;
            }
        }

        private void DrawAudioDropdown()
        {
            GUILayout.Label("Audio");
            selectedAudioIndex = Mathf.Clamp(selectedAudioIndex, 0, AudioOptions.Length - 1);
            if (GUILayout.Button($"{AudioOptions[selectedAudioIndex]} {(_audioListExpanded ? "▲" : "▼")}"))
            {
                _audioListExpanded = !_audioListExpanded;
            }

            if (!_audioListExpanded)
            {
                return;
            }

            for (var i = 0; i < AudioOptions.Length; i++)
            {
                if (GUILayout.Button((i == selectedAudioIndex ? "✓ " : "   ") + AudioOptions[i]))
                {
                    selectedAudioIndex = i;
                    _audioListExpanded = false;
                }
            }
        }

        private AsrModelOption GetSelectedModelOption()
        {
            if (modelOptions == null || modelOptions.Length == 0)
            {
                return null;
            }

            return modelOptions[Mathf.Clamp(selectedModelIndex, 0, modelOptions.Length - 1)];
        }

        private static bool IsOptionEnabled(AsrModelOption option)
        {
            return option != null;
        }

        private static string GetOptionLabel(AsrModelOption option)
        {
            return option == null ? "(unset)" : option.label;
        }

        private IEnumerator TranscribeRoutine(AsrModelOption option)
        {
            _isBusy = true;
            _requestStartedAt = Time.realtimeSinceStartup;
            _status = "Preparing ASR assets...";

            string resolvedModelPath = null;
            string resolvedTokenizerPath = null;
            string resolvedAudioPath = null;
            Exception resolveError = null;

            yield return ResolveStreamingAssetPath(option.modelPath, path => resolvedModelPath = path, ex => resolveError = ex);
            if (resolveError == null)
            {
                yield return ResolveStreamingAssetPath(option.tokenizerJsonPath, path => resolvedTokenizerPath = path, ex => resolveError = ex);
            }

            if (resolveError == null)
            {
                yield return ResolveStreamingAssetPath(AudioOptions[selectedAudioIndex], path => resolvedAudioPath = path, ex => resolveError = ex);
            }

            if (resolveError == null && IsWhisperGpuRequested(option))
            {
                var encoderCompanionPath = GetWhisperEncoderCompanionPath(option.modelPath);
                if (!string.IsNullOrWhiteSpace(encoderCompanionPath))
                {
                    yield return ResolveStreamingAssetPath(encoderCompanionPath, _ => { }, ex => resolveError = ex);
                }
            }

            if (resolveError != null)
            {
                _status = $"Error: {resolveError.Message}";
                Debug.LogException(resolveError);
                _isBusy = false;
                yield break;
            }

            try
            {
                _status = $"Transcribing with {option.label}...";
                var startedAt = Time.realtimeSinceStartup;
                var asrJson = string.Equals(option.mode, QwenAsrMode, StringComparison.OrdinalIgnoreCase)
                    ? _client.RunQwen3AsrSmoke(
                        resolvedModelPath,
                        resolvedAudioPath,
                        resolvedTokenizerPath,
                        backend,
                        language)
                    : _client.RunWhisperAsrSmoke(
                        resolvedModelPath,
                        resolvedAudioPath,
                        resolvedTokenizerPath,
                        backend,
                        language);
                var elapsedSeconds = Time.realtimeSinceStartup - startedAt;

                if (string.IsNullOrWhiteSpace(asrJson))
                {
                    throw new InvalidOperationException("ASR returned an empty result.");
                }

                if (asrJson.Contains("\"success\": false", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidOperationException($"ASR reported failure: {OneLine(Truncate(asrJson, 1200))}");
                }

                var transcript = ExtractTranscript(asrJson);
                var expectedLine = DescribeExpectedMatch(selectedAudioIndex, transcript);
                _transcriptLog.Add(
                    $"model={option.label}, audio={AudioOptions[selectedAudioIndex]}, backend={backend}, language={language}\n" +
                    $"elapsedSeconds={elapsedSeconds:0.###}\n" +
                    $"transcript={transcript}\n" +
                    expectedLine +
                    $"raw={OneLine(Truncate(asrJson, 1800))}");
                _logScroll = new Vector2(0f, float.MaxValue);
                _status = "Transcription complete";
            }
            catch (Exception ex)
            {
                _status = $"Error: {ex.Message}";
                Debug.LogException(ex);
            }

            _isBusy = false;
        }

        private IEnumerator ResolveStreamingAssetPath(
            string configuredPath,
            Action<string> onSuccess,
            Action<Exception> onError)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
            {
                onError(new ArgumentException("Asset path is required."));
                yield break;
            }

            if (Path.IsPathRooted(configuredPath))
            {
                if (!File.Exists(configuredPath))
                {
                    onError(new FileNotFoundException($"File not found: {configuredPath}", configuredPath));
                    yield break;
                }

                onSuccess(configuredPath);
                yield break;
            }

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "ASR");
            Directory.CreateDirectory(destinationDirectory);
            var destinationPath = Path.Combine(destinationDirectory, configuredPath);
            var destinationParent = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(destinationParent))
            {
                Directory.CreateDirectory(destinationParent);
            }

            if (File.Exists(destinationPath))
            {
                onSuccess(destinationPath);
                yield break;
            }

            var sourcePath = Path.Combine(Application.streamingAssetsPath, configuredPath).Replace("\\", "/");
            using var request = UnityWebRequest.Get(sourcePath);
            request.downloadHandler = new DownloadHandlerFile(destinationPath)
            {
                removeFileOnAbort = true,
            };

            yield return request.SendWebRequest();
            if (request.result != UnityWebRequest.Result.Success)
            {
                onError(new IOException($"Failed to copy asset from StreamingAssets: {request.error}"));
                yield break;
            }

            if (!File.Exists(destinationPath) || new FileInfo(destinationPath).Length <= 0)
            {
                onError(new IOException($"Copy produced a missing or empty destination file: {destinationPath}"));
                yield break;
            }

            onSuccess(destinationPath);
        }

        private bool IsWhisperGpuRequested(AsrModelOption option)
        {
            return string.Equals(option.mode, "whisper", StringComparison.OrdinalIgnoreCase) &&
                   !string.IsNullOrWhiteSpace(backend) &&
                   backend.Trim().StartsWith("GPU", StringComparison.OrdinalIgnoreCase);
        }

        private static string GetWhisperEncoderCompanionPath(string configuredModelPath)
        {
            if (string.IsNullOrWhiteSpace(configuredModelPath) ||
                !configuredModelPath.EndsWith(".tflite", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            return configuredModelPath.Substring(0, configuredModelPath.Length - ".tflite".Length) + "_encoder.tflite";
        }

        // Display-only expected-transcript comparison. Spacing, commas and
        // other punctuation are ignored so near-misses like "볼륨업" vs
        // "볼륨 업" still read as a match; the model input is never biased.
        private static string DescribeExpectedMatch(int audioIndex, string transcript)
        {
            if (audioIndex < 0 || audioIndex >= ExpectedTranscripts.Length)
            {
                return string.Empty;
            }

            var expected = ExpectedTranscripts[audioIndex];
            var matched = NormalizeForMatch(expected) == NormalizeForMatch(transcript);
            return $"expected={expected}, expectedMatch={(matched ? "MATCH" : "MISMATCH")} (space/punct-insensitive)\n";
        }

        private static string NormalizeForMatch(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return string.Empty;
            }

            return Regex.Replace(value, "[\\s.,!?\"'()\\-:;·、。，]+", string.Empty)
                .ToLowerInvariant();
        }

        private static string ExtractTranscript(string asrJson)
        {
            foreach (var key in new[] { "transcriptCandidate", "transcript", "text" })
            {
                var match = Regex.Match(
                    asrJson ?? string.Empty,
                    "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"(?<value>(?:\\\\.|[^\"\\\\])*)\"");
                if (match.Success)
                {
                    return Regex.Unescape(match.Groups["value"].Value).Trim();
                }
            }

            return string.Empty;
        }

        private string DrawImeTextField(string value, string controlName)
        {
            GUI.SetNextControlName(controlName);
            var updatedValue = GUILayout.TextField(value ?? string.Empty);
            var rect = GUILayoutUtility.GetLastRect();

            if (string.Equals(GUI.GetNameOfFocusedControl(), controlName, StringComparison.Ordinal))
            {
                _hasImeTextFieldFocus = true;
                Input.imeCompositionMode = IMECompositionMode.On;
                Input.compositionCursorPos = GUIUtility.GUIToScreenPoint(new Vector2(rect.xMax, rect.yMax));
            }

            return updatedValue;
        }

        private static string Truncate(string value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
            {
                return value;
            }

            return value.Substring(0, maxLength) + "...";
        }

        private static string OneLine(string value)
        {
            return string.IsNullOrEmpty(value)
                ? string.Empty
                : value.Replace("\r", " ").Replace("\n", " ").Trim();
        }
    }
}
