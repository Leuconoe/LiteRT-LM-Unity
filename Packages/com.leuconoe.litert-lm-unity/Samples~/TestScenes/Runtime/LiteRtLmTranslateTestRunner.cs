using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Translation test scene runner with two engines:
    ///   1. Whisper Direct — Whisper's native X-to-English translation via the
    ///      &lt;|translate|&gt; decoder task token (output is always English).
    ///   2. ASR + LLM — transcript from any ASR path is fed to an on-device
    ///      LLM with a translation prompt (target language selectable).
    /// Audio source is either a bundled test clip or a single-shot mic
    /// capture (LiteRtLmMicVadCapture).
    /// </summary>
    public sealed class LiteRtLmTranslateTestRunner : MonoBehaviour, ILiteRtLmModelHost
    {
        private const string LogPrefix = "[LiteRT-LM Translate]";
        private const string StatusFileName = "LiteRtLmTranslateTest.status.txt";
        private const string QwenAsrMode = "qwen3-asr";

        private static readonly string[] AudioOptions =
        {
            "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3",
            "TestAssets/Audio/Tactical Evaluation Results Report - March 5, 2025.mp3",
            "TestAssets/Audio/현재 서울의 날씨는 흐림 입니다.mp3",
            "TestAssets/Audio/The current weather in Seoul is cloudy.mp3",
            "TestAssets/Audio/변경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다..mp3",
            "TestAssets/Audio/We are currently reviewing the changes Additional issues. may be discovered during the app review process..mp3",
            "TestAssets/Audio/volume-볼륨 업.mp3",
            "TestAssets/Audio/volume-볼륨, 업.mp3",
            "TestAssets/Audio/volume-소리 키워줘.mp3",
            "TestAssets/Audio/volume-음량 증가.mp3",
        };

        [Serializable]
        public sealed class AsrModelOption
        {
            public string label;
            public string modelPath;
            public string tokenizerJsonPath;
            public string mode;
        }

        // Stock (non-distilled) Whisper tiers only for the direct-translate
        // engine: the ACFT-KO models were distilled on the transcribe task
        // only, so their <|translate|> quality is unvalidated. Qwen3-ASR has
        // no translate task and is usable in the ASR+LLM engine only.
        [Header("LiteRT-LM Translate Test")]
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
                label = "Whisper Base i8 (default)",
                modelPath = "ASR/whisper-base/whisper_base_30s_i8.tflite",
                tokenizerJsonPath = "ASR/whisper-base/tokenizer.json",
                mode = "whisper",
            },
            new AsrModelOption
            {
                label = "Whisper Medium i8",
                modelPath = "ASR/whisper-medium/whisper_medium_30s_i8.tflite",
                tokenizerJsonPath = "ASR/whisper-medium/tokenizer.json",
                mode = "whisper",
            },
            new AsrModelOption
            {
                label = "Whisper Large-v3-Turbo i8 (51866 vocab)",
                modelPath = "ASR/whisper-large-v3-turbo/whisper_large_v3_turbo_30s_i8.tflite",
                tokenizerJsonPath = "ASR/whisper-large-v3-turbo/tokenizer.json",
                mode = "whisper",
            },
            new AsrModelOption
            {
                label = "Qwen3-ASR-0.6B i8 (ASR+LLM only)",
                modelPath = "ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite",
                tokenizerJsonPath = "ASR/qwen3-asr-0.6b/tokenizer.json",
                mode = QwenAsrMode,
            },
        };

        private static readonly string[] EngineLabels = { "Whisper Direct → EN", "ASR + LLM" };
        private static readonly string[] InputModeLabels = { "File", "Mic" };
        private static readonly string[] TargetLanguages = { "English", "Japanese", "Chinese" };

        [SerializeField] private int selectedEngineIndex;
        [SerializeField] private int selectedInputModeIndex;
        [SerializeField] private int selectedModelIndex = 1; // Whisper Base i8
        [SerializeField] private int selectedAudioIndex;
        [SerializeField] private int selectedTargetLanguageIndex;
        [SerializeField] private string asrLanguage = "ko";
        [SerializeField] private string asrBackend = "CPU";
        [SerializeField] private string vadMode = "energy";

        // ASR+LLM engine: translation LLM. Qwen3-0.6B mixed int4 (CPU,
        // ~21 tok/s on device) with " /no_think" appended keeps turns fast.
        [SerializeField] private string llmModelPath = "LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm";
        [SerializeField] private string llmBackend = "CPU";
        [SerializeField] private int llmMaxNumTokens = 1024;

        private LiteRtLmUnityClient _client;
        private LiteRtLmMicVadCapture _micCapture;
        private readonly List<string> _resultLog = new List<string>();
        private string _status = "Idle";
        private bool _isBusy;
        private bool _llmInitialized;
        private string _llmInitializedModelPath;
        private bool _modelListExpanded;
        private bool _audioListExpanded;
        private bool _targetListExpanded;
        private float _requestStartedAt;
        private Vector2 _logScroll;
        private string _lastMicWavPath;

        // Headless verification hook: an optional config file in
        // persistentDataPath auto-drives both engines on startup so devices
        // without touch input can be verified via logcat + status file.
        //   { "runWhisperDirect": true, "runAsrLlm": true,
        //     "audioIndex": 0, "targetLanguage": "English" }
        private const string AutoTestConfigFileName = "LiteRtLmTranslateTest.autotest.json";

        private void Awake()
        {
            _client ??= new LiteRtLmUnityClient();
        }

        private void Start()
        {
            Application.runInBackground = true;
            var configPath = Path.Combine(Application.persistentDataPath, AutoTestConfigFileName);
            if (File.Exists(configPath))
            {
                StartCoroutine(AutoTestRoutine(configPath));
            }
        }

        private void OnDestroy()
        {
            ReleaseModels();
        }

        /// <inheritdoc />
        public void ReleaseModels()
        {
            if (_micCapture != null)
            {
                _micCapture.OnUtteranceCaptured -= HandleMicUtteranceCaptured;
                _micCapture.OnCaptureError -= HandleMicCaptureError;
                _micCapture.Continuous = false;
                if (_micCapture.IsCapturing)
                {
                    _micCapture.StopListening();
                }
            }

            _client?.Dispose();
            _client = null;
        }

        private IEnumerator AutoTestRoutine(string configPath)
        {
            var runWhisperDirect = false;
            var runAsrLlm = false;
            try
            {
                var json = File.ReadAllText(configPath);
                runWhisperDirect = Regex.IsMatch(json, "\"runWhisperDirect\"\\s*:\\s*true");
                runAsrLlm = Regex.IsMatch(json, "\"runAsrLlm\"\\s*:\\s*true");
                var audioMatch = Regex.Match(json, "\"audioIndex\"\\s*:\\s*(?<value>[0-9]+)");
                if (audioMatch.Success)
                {
                    selectedAudioIndex = Mathf.Clamp(
                        int.Parse(audioMatch.Groups["value"].Value), 0, AudioOptions.Length - 1);
                }

                var targetMatch = Regex.Match(json, "\"targetLanguage\"\\s*:\\s*\"(?<value>[^\"]+)\"");
                if (targetMatch.Success)
                {
                    var requested = targetMatch.Groups["value"].Value.Trim();
                    for (var i = 0; i < TargetLanguages.Length; i++)
                    {
                        if (string.Equals(TargetLanguages[i], requested, StringComparison.OrdinalIgnoreCase))
                        {
                            selectedTargetLanguageIndex = i;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                WriteStatus("AUTOTEST_CONFIG_ERROR", ex.Message);
                yield break;
            }

            WriteStatus(
                "AUTOTEST_START",
                $"runWhisperDirect={runWhisperDirect}, runAsrLlm={runAsrLlm}, audioIndex={selectedAudioIndex}, target={TargetLanguages[selectedTargetLanguageIndex]}");
            yield return null;

            if (runWhisperDirect)
            {
                selectedEngineIndex = 0;
                selectedInputModeIndex = 0;
                yield return TranslateFileRoutine();
            }

            if (runAsrLlm)
            {
                while (_isBusy)
                {
                    yield return null;
                }

                selectedEngineIndex = 1;
                selectedInputModeIndex = 0;
                yield return TranslateFileRoutine();
            }

            WriteStatus("AUTOTEST_DONE", $"results={_resultLog.Count}");
        }

        private void OnGUI()
        {
            _client ??= new LiteRtLmUnityClient();

            GUILayout.BeginArea(new Rect(20, 20, 760, 900), GUI.skin.box);
            GUILayout.Label("LiteRT-LM Translate Test");
            GUILayout.Label($"Status: {_status}");
            if (_isBusy)
            {
                GUILayout.Label($"Elapsed: {Time.realtimeSinceStartup - _requestStartedAt:0.0}s");
            }

            if (!_client.IsAvailable)
            {
                GUILayout.Label("Translation requires an Android device build.");
            }

            GUILayout.Label("Engine");
            var newEngineIndex = GUILayout.Toolbar(
                Mathf.Clamp(selectedEngineIndex, 0, EngineLabels.Length - 1), EngineLabels);
            if (newEngineIndex != selectedEngineIndex)
            {
                selectedEngineIndex = newEngineIndex;
                Debug.Log($"{LogPrefix} engine -> {EngineLabels[selectedEngineIndex]}");
                // Whisper Direct cannot use Qwen3-ASR (no translate task).
                if (IsWhisperDirectEngine() && !IsWhisperOption(GetSelectedModelOption()))
                {
                    selectedModelIndex = 1;
                }
            }

            if (IsWhisperDirectEngine())
            {
                GUILayout.Label("Target: English (Whisper native <|translate|> task)");
            }
            else
            {
                DrawTargetLanguageDropdown();
            }

            DrawModelDropdown();

            GUILayout.Label("Input");
            var newInputModeIndex = GUILayout.Toolbar(
                Mathf.Clamp(selectedInputModeIndex, 0, InputModeLabels.Length - 1), InputModeLabels);
            if (newInputModeIndex != selectedInputModeIndex)
            {
                selectedInputModeIndex = newInputModeIndex;
                Debug.Log($"{LogPrefix} input mode -> {InputModeLabels[selectedInputModeIndex]}");
            }

            var micMode = selectedInputModeIndex == 1;
            if (micMode)
            {
                DrawMicSection();
            }
            else
            {
                DrawAudioDropdown();
                GUI.enabled = _client.IsAvailable && !_isBusy;
                if (GUILayout.Button("Translate", GUILayout.Height(40)))
                {
                    Debug.Log($"{LogPrefix} Translate pressed");
                    StartCoroutine(TranslateFileRoutine());
                }

                GUI.enabled = true;
            }

            GUILayout.Space(8);
            GUILayout.Label("Source Transcript / Translation / Timings");
            _logScroll = GUILayout.BeginScrollView(_logScroll, GUILayout.Height(430));
            foreach (var entry in _resultLog)
            {
                GUILayout.TextArea(entry);
                GUILayout.Space(4);
            }

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private bool IsWhisperDirectEngine()
        {
            return selectedEngineIndex == 0;
        }

        private static bool IsWhisperOption(AsrModelOption option)
        {
            return option != null && string.Equals(option.mode, "whisper", StringComparison.OrdinalIgnoreCase);
        }

        private AsrModelOption GetSelectedModelOption()
        {
            if (modelOptions == null || modelOptions.Length == 0)
            {
                return null;
            }

            return modelOptions[Mathf.Clamp(selectedModelIndex, 0, modelOptions.Length - 1)];
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
            var selected = modelOptions[selectedModelIndex];
            if (GUILayout.Button($"{selected.label} {(_modelListExpanded ? "▲" : "▼")}"))
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
                GUI.enabled = !IsWhisperDirectEngine() || IsWhisperOption(option);
                if (GUILayout.Button((i == selectedModelIndex ? "✓ " : "   ") + option.label))
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

        private void DrawTargetLanguageDropdown()
        {
            GUILayout.Label("Target Language");
            selectedTargetLanguageIndex = Mathf.Clamp(selectedTargetLanguageIndex, 0, TargetLanguages.Length - 1);
            if (GUILayout.Button($"{TargetLanguages[selectedTargetLanguageIndex]} {(_targetListExpanded ? "▲" : "▼")}"))
            {
                _targetListExpanded = !_targetListExpanded;
            }

            if (!_targetListExpanded)
            {
                return;
            }

            for (var i = 0; i < TargetLanguages.Length; i++)
            {
                if (GUILayout.Button((i == selectedTargetLanguageIndex ? "✓ " : "   ") + TargetLanguages[i]))
                {
                    selectedTargetLanguageIndex = i;
                    _targetListExpanded = false;
                }
            }
        }

        // Single-shot mic capture: VAD endpoints the utterance, the WAV is
        // saved and translated with the currently selected engine.
        private void DrawMicSection()
        {
            EnsureMicCapture();

            var levelDb = _micCapture.CurrentLevelDb;
            GUILayout.Label(_micCapture.IsCapturing
                ? $"Mic VAD: {_micCapture.State}  level={levelDb:0.0} dB"
                : "Mic VAD: Idle");

            if (!_micCapture.IsCapturing)
            {
                GUI.enabled = !_isBusy;
                if (GUILayout.Button("Listen (speak; VAD auto-stops and translates)", GUILayout.Height(40)))
                {
                    Debug.Log($"{LogPrefix} Listen pressed");
                    _status = "Listening...";
                    _micCapture.Continuous = false;
                    _micCapture.StartListening();
                }

                GUI.enabled = true;
            }
            else if (GUILayout.Button("Stop Listening", GUILayout.Height(40)))
            {
                _micCapture.StopListening();
                _status = "Mic capture cancelled";
            }

            if (!string.IsNullOrEmpty(_lastMicWavPath))
            {
                GUILayout.Label($"Last capture: {_lastMicWavPath}");
            }
        }

        private void EnsureMicCapture()
        {
            if (_micCapture != null)
            {
                return;
            }

            _micCapture = GetComponent<LiteRtLmMicVadCapture>();
            if (_micCapture == null)
            {
                _micCapture = gameObject.AddComponent<LiteRtLmMicVadCapture>();
            }

            _micCapture.OnUtteranceCaptured += HandleMicUtteranceCaptured;
            _micCapture.OnCaptureError += HandleMicCaptureError;
        }

        private void HandleMicCaptureError(string message)
        {
            _status = $"Mic error: {message}";
        }

        private void HandleMicUtteranceCaptured(float[] pcm16k)
        {
            string wavPath;
            try
            {
                var captureDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "MicCaptures");
                Directory.CreateDirectory(captureDirectory);
                wavPath = Path.Combine(captureDirectory, $"mic_{DateTime.Now:yyyyMMdd_HHmmss_fff}.wav");
                LiteRtLmMicVadCapture.WriteWav16BitMono(wavPath, pcm16k, LiteRtLmMicVadCapture.TargetSampleRate);
            }
            catch (Exception ex)
            {
                _status = $"Mic error: failed to write WAV ({ex.Message})";
                Debug.LogException(ex);
                return;
            }

            _lastMicWavPath = wavPath;
            var seconds = pcm16k.Length / (float)LiteRtLmMicVadCapture.TargetSampleRate;
            Debug.Log($"{LogPrefix} mic capture: {seconds:0.00}s endpointed -> {wavPath}");
            if (!_client.IsAvailable)
            {
                _status = "Mic capture saved (translation requires Android build)";
                return;
            }

            if (_isBusy)
            {
                _status = "Mic capture saved (runner busy)";
                return;
            }

            StartCoroutine(TranslateRoutine(wavPath, "mic"));
        }

        private IEnumerator TranslateFileRoutine()
        {
            selectedAudioIndex = Mathf.Clamp(selectedAudioIndex, 0, AudioOptions.Length - 1);
            return TranslateRoutine(AudioOptions[selectedAudioIndex], AudioOptions[selectedAudioIndex]);
        }

        private IEnumerator TranslateRoutine(string audioSourcePath, string audioLabel)
        {
            var option = GetSelectedModelOption();
            if (option == null)
            {
                _status = "No ASR model selected.";
                yield break;
            }

            if (IsWhisperDirectEngine() && !IsWhisperOption(option))
            {
                _status = "Whisper Direct engine requires a Whisper model.";
                yield break;
            }

            _isBusy = true;
            _requestStartedAt = Time.realtimeSinceStartup;
            _status = "Preparing assets...";

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
                yield return ResolveStreamingAssetPath(audioSourcePath, path => resolvedAudioPath = path, ex => resolveError = ex);
            }

            if (resolveError != null)
            {
                _status = $"Error: {resolveError.Message}";
                WriteStatus("FAILURE", resolveError.ToString());
                Debug.LogException(resolveError);
                _isBusy = false;
                yield break;
            }

            if (IsWhisperDirectEngine())
            {
                yield return WhisperDirectRoutine(option, resolvedModelPath, resolvedTokenizerPath, resolvedAudioPath, audioLabel);
            }
            else
            {
                yield return AsrLlmRoutine(option, resolvedModelPath, resolvedTokenizerPath, resolvedAudioPath, audioLabel);
            }

            _isBusy = false;
        }

        // Engine 1: two Whisper passes over the same clip — task=transcribe
        // (source-language transcript) then task=translate (English). The
        // native compiled-model cache makes the second pass compile-free.
        private IEnumerator WhisperDirectRoutine(
            AsrModelOption option,
            string resolvedModelPath,
            string resolvedTokenizerPath,
            string resolvedAudioPath,
            string audioLabel)
        {
            _status = $"Whisper direct translate ({option.label})...";
            _client.WarmUpBridge();

            var transcribeStartedAt = Time.realtimeSinceStartup;
            var transcribeTask = System.Threading.Tasks.Task.Run(() =>
                RunWhisperOnWorkerThread(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath, "transcribe"));
            while (!transcribeTask.IsCompleted)
            {
                yield return null;
            }

            var transcribeSeconds = Time.realtimeSinceStartup - transcribeStartedAt;
            if (!TryGetTaskResult(transcribeTask, "whisper transcribe", out var transcribeJson))
            {
                yield break;
            }

            var sourceTranscript = ExtractTranscript(transcribeJson);

            _status = "Whisper translate pass...";
            var translateStartedAt = Time.realtimeSinceStartup;
            var translateTask = System.Threading.Tasks.Task.Run(() =>
                RunWhisperOnWorkerThread(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath, "translate"));
            while (!translateTask.IsCompleted)
            {
                yield return null;
            }

            var translateSeconds = Time.realtimeSinceStartup - translateStartedAt;
            if (!TryGetTaskResult(translateTask, "whisper translate", out var translateJson))
            {
                yield break;
            }

            var translation = ExtractTranscript(translateJson);
            var taskTokenId = ExtractJsonNumber(translateJson, "taskTokenId");
            var entry =
                $"engine=WhisperDirect, model={option.label}, audio={audioLabel}, backend={asrBackend}, language={asrLanguage}\n" +
                $"source transcript ({asrLanguage}): {sourceTranscript}\n" +
                $"translation (English): {translation}\n" +
                $"timings: transcribe={transcribeSeconds:0.###}s, translate={translateSeconds:0.###}s, total={transcribeSeconds + translateSeconds:0.###}s" +
                (float.IsNaN(taskTokenId) ? string.Empty : $", taskTokenId={taskTokenId:0}");
            _resultLog.Add(entry);
            _logScroll = new Vector2(0f, float.MaxValue);
            _status = "Whisper direct translation complete";
            WriteStatus("WHISPER_DIRECT_RESULT", OneLine(entry));
        }

        // Engine 2: ASR transcript -> LLM translation prompt.
        private IEnumerator AsrLlmRoutine(
            AsrModelOption option,
            string resolvedModelPath,
            string resolvedTokenizerPath,
            string resolvedAudioPath,
            string audioLabel)
        {
            _status = $"Transcribing with {option.label}...";
            _client.WarmUpBridge();

            var asrStartedAt = Time.realtimeSinceStartup;
            var isQwenAsr = string.Equals(option.mode, QwenAsrMode, StringComparison.OrdinalIgnoreCase);
            var asrTask = System.Threading.Tasks.Task.Run(() =>
                isQwenAsr
                    ? RunQwen3AsrOnWorkerThread(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath)
                    : RunWhisperOnWorkerThread(resolvedModelPath, resolvedAudioPath, resolvedTokenizerPath, "transcribe"));
            while (!asrTask.IsCompleted)
            {
                yield return null;
            }

            var asrSeconds = Time.realtimeSinceStartup - asrStartedAt;
            if (!TryGetTaskResult(asrTask, "ASR", out var asrJson))
            {
                yield break;
            }

            var transcript = ExtractTranscript(asrJson);
            if (string.IsNullOrWhiteSpace(transcript))
            {
                _status = "ASR produced an empty transcript.";
                WriteStatus("FAILURE", $"empty transcript, raw={OneLine(Truncate(asrJson, 800))}");
                yield break;
            }

            string resolvedLlmModelPath = null;
            Exception resolveError = null;
            yield return ResolveStreamingAssetPath(llmModelPath, path => resolvedLlmModelPath = path, ex => resolveError = ex);
            if (resolveError != null)
            {
                _status = $"Error: {resolveError.Message}";
                WriteStatus("FAILURE", resolveError.ToString());
                yield break;
            }

            var llmInitSeconds = 0f;
            if (!_llmInitialized || !string.Equals(_llmInitializedModelPath, resolvedLlmModelPath, StringComparison.Ordinal))
            {
                _status = "Initializing translation LLM...";
                yield return null; // let the status paint before the blocking init
                var initStartedAt = Time.realtimeSinceStartup;
                Exception initError = null;
                try
                {
                    // Recreate the client so a previous LLM session (or the
                    // ASR-only bridge) does not leak into this initialization
                    // (same pattern as the ASR function-calling demo).
                    _client.Dispose();
                    _client = new LiteRtLmUnityClient();
                    _client.WarmUpBridge();

                    _client.Initialize(
                        resolvedLlmModelPath,
                        llmBackend,
                        Application.temporaryCachePath,
                        llmMaxNumTokens,
                        0,
                        0,
                        false,
                        BuildTranslationSystemInstruction());
                    _llmInitialized = true;
                    _llmInitializedModelPath = resolvedLlmModelPath;
                }
                catch (Exception ex)
                {
                    initError = ex;
                }

                llmInitSeconds = Time.realtimeSinceStartup - initStartedAt;
                if (initError != null)
                {
                    _status = $"LLM init error: {initError.Message}";
                    WriteStatus("FAILURE", initError.ToString());
                    Debug.LogException(initError);
                    yield break;
                }

                WriteStatus("LLM_INITIALIZED", $"model={llmModelPath}, backend={llmBackend}, elapsedSeconds={llmInitSeconds:0.###}");
            }

            var targetLanguage = TargetLanguages[Mathf.Clamp(selectedTargetLanguageIndex, 0, TargetLanguages.Length - 1)];
            var prompt = BuildTranslationPrompt(transcript, targetLanguage);
            _status = $"Translating to {targetLanguage} with LLM...";
            WriteStatus("LLM_PROMPT", OneLine(prompt));

            var llmStartedAt = Time.realtimeSinceStartup;
            var llmTask = System.Threading.Tasks.Task.Run(() => RunLlmOnWorkerThread(prompt));
            while (!llmTask.IsCompleted)
            {
                yield return null;
            }

            var llmSeconds = Time.realtimeSinceStartup - llmStartedAt;
            if (llmTask.IsFaulted || llmTask.IsCanceled)
            {
                var error = llmTask.Exception?.GetBaseException();
                _status = $"LLM error: {error?.Message ?? "canceled"}";
                WriteStatus("FAILURE", error?.ToString() ?? "LLM task canceled");
                yield break;
            }

            var translation = CleanLlmTranslation(llmTask.Result);
            if (string.IsNullOrWhiteSpace(translation))
            {
                _status = "LLM returned an empty translation.";
                WriteStatus("FAILURE", $"empty translation, raw={OneLine(Truncate(llmTask.Result, 800))}");
                yield break;
            }

            var entry =
                $"engine=ASR+LLM, asrModel={option.label}, llm={llmModelPath} ({llmBackend}), audio={audioLabel}, target={targetLanguage}\n" +
                $"source transcript ({asrLanguage}): {transcript}\n" +
                $"translation ({targetLanguage}): {translation}\n" +
                $"timings: asr={asrSeconds:0.###}s" +
                (llmInitSeconds > 0f ? $", llmInit={llmInitSeconds:0.###}s" : string.Empty) +
                $", llm={llmSeconds:0.###}s, total={asrSeconds + llmInitSeconds + llmSeconds:0.###}s";
            _resultLog.Add(entry);
            _logScroll = new Vector2(0f, float.MaxValue);
            _status = "ASR+LLM translation complete";
            WriteStatus("ASR_LLM_RESULT", OneLine(entry));
        }

        private bool TryGetTaskResult(System.Threading.Tasks.Task<string> task, string label, out string json)
        {
            json = null;
            if (task.IsFaulted || task.IsCanceled)
            {
                var error = task.Exception?.GetBaseException();
                _status = $"{label} error: {error?.Message ?? "canceled"}";
                WriteStatus("FAILURE", error?.ToString() ?? $"{label} task canceled");
                return false;
            }

            json = task.Result;
            if (string.IsNullOrWhiteSpace(json))
            {
                _status = $"{label} returned an empty result.";
                WriteStatus("FAILURE", $"{label} returned an empty result.");
                return false;
            }

            if (json.Contains("\"success\": false", StringComparison.OrdinalIgnoreCase))
            {
                _status = $"{label} reported failure.";
                WriteStatus("FAILURE", $"{label} failed: {OneLine(Truncate(json, 1000))}");
                return false;
            }

            return true;
        }

        // Blocking native calls run on Task threads so the IMGUI stays live;
        // the Java bridge object is created up front on the main thread by
        // WarmUpBridge() and only its methods are invoked here.
        private string RunWhisperOnWorkerThread(string modelPath, string audioPath, string tokenizerPath, string task)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            AndroidJNI.AttachCurrentThread();
            try
            {
#endif
                return _client.RunWhisperAsrSmoke(
                    modelPath, audioPath, tokenizerPath, asrBackend, asrLanguage, vadMode, string.Empty, task);
#if UNITY_ANDROID && !UNITY_EDITOR
            }
            finally
            {
                AndroidJNI.DetachCurrentThread();
            }
#endif
        }

        private string RunQwen3AsrOnWorkerThread(string modelPath, string audioPath, string tokenizerPath)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            AndroidJNI.AttachCurrentThread();
            try
            {
#endif
                return _client.RunQwen3AsrSmoke(modelPath, audioPath, tokenizerPath, asrBackend, asrLanguage, vadMode);
#if UNITY_ANDROID && !UNITY_EDITOR
            }
            finally
            {
                AndroidJNI.DetachCurrentThread();
            }
#endif
        }

        private string RunLlmOnWorkerThread(string prompt)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            AndroidJNI.AttachCurrentThread();
            try
            {
#endif
                return _client.SendMessage(prompt);
#if UNITY_ANDROID && !UNITY_EDITOR
            }
            finally
            {
                AndroidJNI.DetachCurrentThread();
            }
#endif
        }

        private static string BuildTranslationSystemInstruction()
        {
            return "You are a translation engine. Translate the user's text exactly. " +
                   "Output only the translation with no explanations, notes or quotes.";
        }

        private string BuildTranslationPrompt(string transcript, string targetLanguage)
        {
            var sourceLanguageName = string.Equals(asrLanguage, "ko", StringComparison.OrdinalIgnoreCase)
                ? "Korean "
                : string.Equals(asrLanguage, "en", StringComparison.OrdinalIgnoreCase) ? "English " : string.Empty;
            // " /no_think" keeps Qwen3 in the fast non-thinking mode; other
            // models ignore the soft switch.
            return $"Translate the following {sourceLanguageName}text to {targetLanguage}. " +
                   "Output only the translation.\n\n" +
                   transcript.Trim() +
                   " /no_think";
        }

        // Strips Qwen3 <think> blocks, the CLI BenchmarkInfo trailer and
        // wrapping quotes from an LLM translation response.
        private static string CleanLlmTranslation(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                return string.Empty;
            }

            var cleaned = raw;
            var benchmarkIndex = cleaned.IndexOf("BenchmarkInfo:", StringComparison.Ordinal);
            if (benchmarkIndex >= 0)
            {
                cleaned = cleaned.Substring(0, benchmarkIndex);
            }

            cleaned = Regex.Replace(cleaned, "<think>[\\s\\S]*?</think>", string.Empty);
            cleaned = cleaned.Trim();
            if (cleaned.Length >= 2 && cleaned[0] == '"' && cleaned[cleaned.Length - 1] == '"')
            {
                cleaned = cleaned.Substring(1, cleaned.Length - 2).Trim();
            }

            return cleaned;
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

        private static float ExtractJsonNumber(string json, string key)
        {
            var match = Regex.Match(
                json ?? string.Empty,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*(?<value>-?[0-9.eE+]+)");
            if (match.Success &&
                float.TryParse(
                    match.Groups["value"].Value,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out var value))
            {
                return value;
            }

            return float.NaN;
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

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "Translate");
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

        private static void WriteStatus(string phase, string message)
        {
            var line = $"{LogPrefix} {phase}: {message}";
            Debug.Log(line);

            try
            {
                var statusPath = Path.Combine(Application.persistentDataPath, StatusFileName);
                File.AppendAllText(statusPath, $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"{LogPrefix} failed to write status file: {ex.Message}");
            }
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
