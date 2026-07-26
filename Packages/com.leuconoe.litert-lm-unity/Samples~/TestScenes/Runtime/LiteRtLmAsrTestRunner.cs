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

        // ASR preprocessing VAD modes (native dual-mode VAD, task #24):
        // Off = no trim/normalization, Energy = adaptive energy gate
        // (default), AI = Silero VAD tflite (falls back to Energy with a
        // vadError field in the result JSON when the model is unavailable).
        private static readonly string[] VadModeOptions = { "off", "energy", "ai" };
        private static readonly string[] VadModeLabels = { "Off", "Energy (default)", "AI (Silero)" };
        private const string SileroVadModelPath = "ASR/silero-vad/silero_vad_16k.tflite";

        // Input source: File = bundled test clip dropdown, Mic = live
        // microphone with C#-side VAD endpointing (LiteRtLmMicVadCapture).
        // Mic captures are already endpointed, so the native vadMode above
        // can stay "energy" — the native trim only re-trims the edges of the
        // captured utterance and the two trims compose safely.
        private static readonly string[] InputModeLabels = { "File", "Mic" };
        private const string MicAudioLabel = "mic";

        [SerializeField] private int selectedInputModeIndex;
        [SerializeField] private int selectedModelIndex;
        [SerializeField] private int selectedAudioIndex;
        [SerializeField] private int selectedVadModeIndex = 1;
        [SerializeField] private string language = "ko";
        [SerializeField] private string backend = "CPU";
        // Mic-mode "Continuous" toggle: one press starts an always-listening
        // loop (VAD endpoints -> bounded queue -> async transcription ->
        // resume listening immediately).
        [SerializeField] private bool continuousMic;

        // A mic utterance endpointed during a continuous session, waiting
        // for the transcription worker. Queue and worker both run on the
        // main thread (event handler + coroutine), so no locking is needed;
        // only the blocking native ASR call itself runs on a Task thread.
        private sealed class PendingUtterance
        {
            public int Index;
            public string WavPath;
            public float Seconds;
            public float EndpointedAt;
        }

        private const int MaxQueuedUtterances = 4;
        private readonly Queue<PendingUtterance> _utteranceQueue = new Queue<PendingUtterance>();
        private bool _continuousActive;
        private bool _continuousStopRequested;
        private int _sessionUtteranceCount;
        private int _sessionTranscribedCount;
        private int _sessionDroppedCount;
        private float _lastUtteranceLatencySeconds = -1f;

        private LiteRtLmUnityClient _client;
        private LiteRtLmMicVadCapture _micCapture;
        private string _lastMicWavPath;
        private readonly List<string> _transcriptLog = new List<string>();
        private string _status = "Idle";
        private bool _isBusy;
        private bool _modelListExpanded;
        private bool _audioListExpanded;
        private bool _vadModeListExpanded;
        private float _requestStartedAt;
        private Vector2 _logScroll;
        private bool _hasImeTextFieldFocus;
        // Control rects logged once per input mode so headless device runs
        // (screencap-blind hardware) can drive the IMGUI via adb input taps.
        private Rect _inputToolbarRect;
        private Rect _actionButtonRect;
        private bool _controlRectsLogged;

        private void Awake()
        {
            _client ??= new LiteRtLmUnityClient();
        }

        // Headless verification hook: devices without a touchscreen (or with
        // FLAG_SECURE displays) cannot be driven via adb taps/screencap, so
        // an optional config file in persistentDataPath auto-drives the
        // scene. Absent file = no behavior change.
        //   { "micSmokeSeconds": 5, "continuousSeconds": 30,
        //     "continuousPlaybackAudioIndex": 0, "fileTranscribe": true }
        // continuousSeconds > 0 runs the always-listening loop for that long
        // (ambient audio above the VAD gate cycles utterances via the 8 s
        // max-utterance cutoff on units where nobody can speak into the mic).
        // continuousPlaybackAudioIndex >= 0 additionally loops that bundled
        // AudioOptions clip through the device speaker during the continuous
        // window, so the microphone hears deterministic speech even in a
        // quiet room (speaker -> mic echo injection).
        private const string AutoTestConfigFileName = "LiteRtLmAsrTest.autotest.json";

        private void Start()
        {
            var configPath = Path.Combine(Application.persistentDataPath, AutoTestConfigFileName);
            if (File.Exists(configPath))
            {
                StartCoroutine(AutoTestRoutine(configPath));
            }
        }

        private IEnumerator AutoTestRoutine(string configPath)
        {
            var micSmokeSeconds = 0f;
            var continuousSeconds = 0f;
            var continuousPlaybackAudioIndex = -1;
            var fileTranscribe = false;
            try
            {
                var json = File.ReadAllText(configPath);
                var micMatch = Regex.Match(json, "\"micSmokeSeconds\"\\s*:\\s*(?<value>[0-9.]+)");
                if (micMatch.Success)
                {
                    micSmokeSeconds = float.Parse(
                        micMatch.Groups["value"].Value,
                        System.Globalization.CultureInfo.InvariantCulture);
                }

                var continuousMatch = Regex.Match(json, "\"continuousSeconds\"\\s*:\\s*(?<value>[0-9.]+)");
                if (continuousMatch.Success)
                {
                    continuousSeconds = float.Parse(
                        continuousMatch.Groups["value"].Value,
                        System.Globalization.CultureInfo.InvariantCulture);
                }

                var playbackMatch = Regex.Match(json, "\"continuousPlaybackAudioIndex\"\\s*:\\s*(?<value>-?[0-9]+)");
                if (playbackMatch.Success)
                {
                    continuousPlaybackAudioIndex = int.Parse(
                        playbackMatch.Groups["value"].Value,
                        System.Globalization.CultureInfo.InvariantCulture);
                }

                fileTranscribe = Regex.IsMatch(json, "\"fileTranscribe\"\\s*:\\s*true");
            }
            catch (Exception ex)
            {
                Debug.LogError($"LiteRtLmAsrTestRunner autotest: failed to read config ({ex.Message})");
                yield break;
            }

            Debug.Log($"LiteRtLmAsrTestRunner autotest: micSmokeSeconds={micSmokeSeconds}, continuousSeconds={continuousSeconds}, continuousPlaybackAudioIndex={continuousPlaybackAudioIndex}, fileTranscribe={fileTranscribe}");
            yield return null;

            if (micSmokeSeconds > 0f)
            {
                selectedInputModeIndex = 1;
                EnsureMicCapture();
                _micCapture.StartListening();
                var deadline = Time.realtimeSinceStartup + micSmokeSeconds;
                while (Time.realtimeSinceStartup < deadline && _micCapture.IsCapturing)
                {
                    yield return null;
                }

                // Ambient noise may have endpointed (and stopped) the capture
                // already; otherwise stop it explicitly to exercise
                // Listening -> Idle.
                if (_micCapture.IsCapturing)
                {
                    _micCapture.StopListening();
                }

                Debug.Log("LiteRtLmAsrTestRunner autotest: mic smoke complete");
            }

            if (continuousSeconds > 0f)
            {
                // A single-shot mic capture endpointed by the smoke phase may
                // still be transcribing — let it settle first.
                while (_isBusy)
                {
                    yield return null;
                }

                // Optional speaker->mic echo injection: loop a bundled clip
                // through the device speaker so the VAD hears deterministic
                // speech even in a quiet room (headless verification units).
                AudioSource playbackSource = null;
                if (continuousPlaybackAudioIndex >= 0 && continuousPlaybackAudioIndex < AudioOptions.Length)
                {
                    string playbackClipPath = null;
                    Exception playbackError = null;
                    yield return ResolveStreamingAssetPath(
                        AudioOptions[continuousPlaybackAudioIndex],
                        path => playbackClipPath = path,
                        ex => playbackError = ex);
                    if (playbackError == null)
                    {
                        using var clipRequest = UnityWebRequestMultimedia.GetAudioClip(
                            "file://" + playbackClipPath, AudioType.MPEG);
                        yield return clipRequest.SendWebRequest();
                        if (clipRequest.result == UnityWebRequest.Result.Success)
                        {
                            var playbackClip = DownloadHandlerAudioClip.GetContent(clipRequest);
                            playbackSource = gameObject.AddComponent<AudioSource>();
                            playbackSource.clip = playbackClip;
                            playbackSource.loop = true;
                            playbackSource.volume = 1f;
                            playbackSource.Play();
                            Debug.Log($"LiteRtLmAsrTestRunner autotest: continuous playback started ({AudioOptions[continuousPlaybackAudioIndex]}, {playbackClip.length:0.0}s loop)");
                        }
                        else
                        {
                            Debug.LogWarning($"LiteRtLmAsrTestRunner autotest: playback clip decode failed ({clipRequest.error})");
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"LiteRtLmAsrTestRunner autotest: playback clip resolve failed ({playbackError.Message})");
                    }
                }

                selectedInputModeIndex = 1;
                continuousMic = true;
                StartContinuousSession(GetSelectedModelOption());
                var continuousDeadline = Time.realtimeSinceStartup + continuousSeconds;
                while (Time.realtimeSinceStartup < continuousDeadline)
                {
                    yield return null;
                }

                StopContinuousSession();
                while (_continuousActive)
                {
                    yield return null;
                }

                if (playbackSource != null)
                {
                    playbackSource.Stop();
                    Destroy(playbackSource);
                    Debug.Log("LiteRtLmAsrTestRunner autotest: continuous playback stopped");
                }

                Debug.Log(
                    "LiteRtLmAsrTestRunner autotest: continuous complete " +
                    $"(captured={_sessionUtteranceCount}, transcribed={_sessionTranscribedCount}, dropped={_sessionDroppedCount})");
            }

            if (fileTranscribe)
            {
                while (_isBusy)
                {
                    yield return null;
                }

                selectedInputModeIndex = 0;
                var option = GetSelectedModelOption();
                if (option != null && _client.IsAvailable)
                {
                    yield return TranscribeRoutine(option);
                }
                else
                {
                    Debug.LogWarning("LiteRtLmAsrTestRunner autotest: file transcribe skipped (no model or client unavailable)");
                }

                Debug.Log("LiteRtLmAsrTestRunner autotest: file transcribe complete");
            }

            Debug.Log("LiteRtLmAsrTestRunner autotest: done");
        }

        private void OnDestroy()
        {
            if (_micCapture != null)
            {
                _micCapture.OnUtteranceCaptured -= HandleMicUtteranceCaptured;
                _micCapture.OnCaptureError -= HandleMicCaptureError;
            }

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

            GUILayout.Label("Input");
            var newInputModeIndex = GUILayout.Toolbar(
                Mathf.Clamp(selectedInputModeIndex, 0, InputModeLabels.Length - 1),
                InputModeLabels);
            if (Event.current.type == EventType.Repaint)
            {
                _inputToolbarRect = GUILayoutUtility.GetLastRect();
            }

            if (newInputModeIndex != selectedInputModeIndex)
            {
                selectedInputModeIndex = newInputModeIndex;
                _controlRectsLogged = false;
                Debug.Log($"LiteRtLmAsrTestRunner input mode -> {InputModeLabels[selectedInputModeIndex]}");
            }

            var micMode = selectedInputModeIndex == 1;
            if (!micMode)
            {
                DrawAudioDropdown();
            }

            DrawVadModeDropdown();

            GUILayout.Label("Language");
            language = DrawImeTextField(language, "LiteRtLmAsrLanguageField");
            GUILayout.Label("Backend");
            backend = DrawImeTextField(backend, "LiteRtLmAsrBackendField");
            if (!_hasImeTextFieldFocus)
            {
                Input.imeCompositionMode = IMECompositionMode.Auto;
            }

            var selectedOption = GetSelectedModelOption();
            if (micMode)
            {
                DrawMicSection(selectedOption);
            }
            else
            {
                var transcribeEnabled = _client.IsAvailable && !_isBusy && !_continuousActive && selectedOption != null && IsOptionEnabled(selectedOption);
                GUI.enabled = transcribeEnabled;
                if (GUILayout.Button("Transcribe", GUILayout.Height(40)))
                {
                    Debug.Log("LiteRtLmAsrTestRunner: Transcribe pressed");
                    StartCoroutine(TranscribeRoutine(selectedOption));
                }

                CaptureActionButtonRect();
                GUI.enabled = true;
            }

            LogControlRectsOnce();

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

        private void CaptureActionButtonRect()
        {
            if (Event.current.type == EventType.Repaint)
            {
                _actionButtonRect = GUILayoutUtility.GetLastRect();
            }
        }

        // Logs the tap targets once per input mode: coordinates are screen
        // pixels (the 20,20 GUILayout.BeginArea origin is added) so headless
        // device verification can use `adb shell input tap` directly.
        private void LogControlRectsOnce()
        {
            if (_controlRectsLogged || Event.current.type != EventType.Repaint)
            {
                return;
            }

            _controlRectsLogged = true;
            Debug.Log(
                "LiteRtLmAsrTestRunner controls: " +
                $"inputToolbar={FormatScreenRect(_inputToolbarRect)}, " +
                $"actionButton={FormatScreenRect(_actionButtonRect)}, " +
                $"mode={InputModeLabels[Mathf.Clamp(selectedInputModeIndex, 0, InputModeLabels.Length - 1)]}");
        }

        private static string FormatScreenRect(Rect rect)
        {
            return $"x={rect.x + 20f:0},y={rect.y + 20f:0},w={rect.width:0},h={rect.height:0}";
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

        private void DrawVadModeDropdown()
        {
            GUILayout.Label("VAD Mode");
            selectedVadModeIndex = Mathf.Clamp(selectedVadModeIndex, 0, VadModeOptions.Length - 1);
            if (GUILayout.Button($"{VadModeLabels[selectedVadModeIndex]} {(_vadModeListExpanded ? "▲" : "▼")}"))
            {
                _vadModeListExpanded = !_vadModeListExpanded;
            }

            if (!_vadModeListExpanded)
            {
                return;
            }

            for (var i = 0; i < VadModeOptions.Length; i++)
            {
                if (GUILayout.Button((i == selectedVadModeIndex ? "✓ " : "   ") + VadModeLabels[i]))
                {
                    selectedVadModeIndex = i;
                    _vadModeListExpanded = false;
                }
            }
        }

        // Live-mic input: level meter + VAD state, endpointed by
        // LiteRtLmMicVadCapture. Capture itself works everywhere (including
        // the Windows editor, for VAD tuning); the transcription step still
        // requires the Android bridge like file mode.
        private void DrawMicSection(AsrModelOption selectedOption)
        {
            EnsureMicCapture();

            var levelDb = _micCapture.CurrentLevelDb;
            var stateLine = _micCapture.IsCapturing
                ? $"Mic VAD: {_micCapture.State}  level={levelDb:0.0} dB  noiseFloor={_micCapture.NoiseFloorDb:0.0} dB  speechOn={_micCapture.SpeechOnThresholdDb:0.0} dB"
                : "Mic VAD: Idle";
            if (_micCapture.State == LiteRtLmMicVadCapture.MicVadState.Speech)
            {
                stateLine += $"  utterance={_micCapture.UtteranceSeconds:0.0}s";
            }

            GUILayout.Label(stateLine);
            DrawMicLevelMeter(levelDb);

            if (_continuousActive)
            {
                GUILayout.Label(
                    $"Continuous session: utterances={_sessionUtteranceCount}  " +
                    $"transcribed={_sessionTranscribedCount}  queue={_utteranceQueue.Count}/{MaxQueuedUtterances}  " +
                    $"dropped={_sessionDroppedCount}" +
                    (_lastUtteranceLatencySeconds >= 0f
                        ? $"  lastLatency={_lastUtteranceLatencySeconds:0.00}s"
                        : string.Empty));

                GUI.enabled = !_continuousStopRequested;
                if (GUILayout.Button("Stop Continuous Listening", GUILayout.Height(40)))
                {
                    Debug.Log("LiteRtLmAsrTestRunner: Stop Continuous pressed");
                    StopContinuousSession();
                }

                CaptureActionButtonRect();
                GUI.enabled = true;
            }
            else if (!_micCapture.IsCapturing)
            {
                continuousMic = GUILayout.Toggle(continuousMic, "Continuous (always listening)");
                GUI.enabled = !_isBusy;
                var listenLabel = continuousMic
                    ? "Start Continuous Listening"
                    : "Listen (speak; VAD auto-stops and transcribes)";
                if (GUILayout.Button(listenLabel, GUILayout.Height(40)))
                {
                    if (continuousMic)
                    {
                        Debug.Log("LiteRtLmAsrTestRunner: Start Continuous pressed");
                        StartContinuousSession(selectedOption);
                    }
                    else
                    {
                        Debug.Log("LiteRtLmAsrTestRunner: Listen pressed");
                        _status = "Listening...";
                        _micCapture.Continuous = false;
                        _micCapture.StartListening();
                    }
                }

                CaptureActionButtonRect();
                GUI.enabled = true;
            }
            else
            {
                if (GUILayout.Button("Stop Listening", GUILayout.Height(40)))
                {
                    Debug.Log("LiteRtLmAsrTestRunner: Stop Listening pressed");
                    _micCapture.StopListening();
                    _status = "Mic capture cancelled";
                }

                CaptureActionButtonRect();
            }

            if (!string.IsNullOrEmpty(_lastMicWavPath))
            {
                GUILayout.Label($"Last capture: {_lastMicWavPath}");
            }

            if (selectedOption == null)
            {
                GUILayout.Label("No ASR model selected — captures will be saved but not transcribed.");
            }
        }

        private void DrawMicLevelMeter(float levelDb)
        {
            var rect = GUILayoutUtility.GetRect(0f, 20f, GUILayout.ExpandWidth(true));
            GUI.Box(rect, GUIContent.none);
            var previousColor = GUI.color;

            var normalizedLevel = Mathf.InverseLerp(-60f, 0f, levelDb);
            if (_micCapture.IsCapturing && normalizedLevel > 0f)
            {
                GUI.color = _micCapture.State == LiteRtLmMicVadCapture.MicVadState.Speech
                    ? new Color(0.3f, 0.9f, 0.3f, 0.9f)
                    : new Color(0.6f, 0.6f, 0.6f, 0.9f);
                GUI.DrawTexture(
                    new Rect(rect.x + 1f, rect.y + 1f, (rect.width - 2f) * normalizedLevel, rect.height - 2f),
                    Texture2D.whiteTexture);
            }

            if (_micCapture.IsCapturing)
            {
                // Speech-on threshold tick.
                var normalizedThreshold = Mathf.InverseLerp(-60f, 0f, _micCapture.SpeechOnThresholdDb);
                GUI.color = new Color(0.95f, 0.35f, 0.25f, 0.9f);
                GUI.DrawTexture(
                    new Rect(rect.x + 1f + (rect.width - 4f) * normalizedThreshold, rect.y + 1f, 2f, rect.height - 2f),
                    Texture2D.whiteTexture);
            }

            GUI.color = previousColor;
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
            if (_continuousActive)
            {
                HandleContinuousUtteranceCaptured(pcm16k);
                return;
            }

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
            var utteranceSeconds = pcm16k.Length / (float)LiteRtLmMicVadCapture.TargetSampleRate;
            var captureLine = $"mic capture: {utteranceSeconds:0.00}s endpointed -> {wavPath}";
            Debug.Log($"LiteRtLmAsrTestRunner {captureLine}");
            _transcriptLog.Add(captureLine);
            _logScroll = new Vector2(0f, float.MaxValue);

            var option = GetSelectedModelOption();
            if (!_client.IsAvailable)
            {
                _status = "Mic capture saved (ASR requires Android build)";
                return;
            }

            if (_isBusy || option == null)
            {
                _status = "Mic capture saved (runner busy or no model selected)";
                return;
            }

            StartCoroutine(TranscribeRoutine(option, wavPath, MicAudioLabel));
        }

        // ------------------------------------------------------------------
        // Continuous (always-listening) session
        //
        // Producer: LiteRtLmMicVadCapture in Continuous mode endpoints
        // utterances and this handler writes each one to WAV and enqueues it
        // (bounded queue, drop-oldest). Consumer: ContinuousWorkerRoutine
        // dequeues and runs the blocking native ASR call on a Task thread so
        // capture never blocks on ASR. Both queue ends run on the main
        // thread, so no locking is needed.
        // ------------------------------------------------------------------

        private void StartContinuousSession(AsrModelOption option)
        {
            if (_continuousActive)
            {
                return;
            }

            _continuousActive = true;
            _continuousStopRequested = false;
            _utteranceQueue.Clear();
            _sessionUtteranceCount = 0;
            _sessionTranscribedCount = 0;
            _sessionDroppedCount = 0;
            _lastUtteranceLatencySeconds = -1f;
            _status = "Continuous listening...";

            // Create the Java bridge on the main thread now; the worker's
            // Task thread only calls methods on the existing object.
            _client.WarmUpBridge();

            EnsureMicCapture();
            _micCapture.Continuous = true;
            _micCapture.StartListening();
            StartCoroutine(ContinuousWorkerRoutine(option));
            Debug.Log($"LiteRtLmAsrTestRunner continuous: session started (model={GetOptionLabel(option)}, queueCapacity={MaxQueuedUtterances})");
        }

        private void StopContinuousSession()
        {
            if (!_continuousActive || _continuousStopRequested)
            {
                return;
            }

            _continuousStopRequested = true;
            _micCapture.Continuous = false;
            _micCapture.StopListening();

            // Stop policy: discard anything still queued (log the count); an
            // in-flight transcription is allowed to finish and log.
            var discarded = _utteranceQueue.Count;
            _utteranceQueue.Clear();
            if (discarded > 0)
            {
                _sessionDroppedCount += discarded;
                Debug.Log($"LiteRtLmAsrTestRunner continuous: stop requested, discarded {discarded} queued utterance(s)");
            }
            else
            {
                Debug.Log("LiteRtLmAsrTestRunner continuous: stop requested");
            }

            _status = "Continuous session stopping...";
        }

        private void HandleContinuousUtteranceCaptured(float[] pcm16k)
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
            _sessionUtteranceCount++;
            var index = _sessionUtteranceCount;
            var utteranceSeconds = pcm16k.Length / (float)LiteRtLmMicVadCapture.TargetSampleRate;

            if (_utteranceQueue.Count >= MaxQueuedUtterances)
            {
                var dropped = _utteranceQueue.Dequeue();
                _sessionDroppedCount++;
                var dropLine = $"[{DateTime.Now:HH:mm:ss}] #{dropped.Index} DROPPED (queue full at {MaxQueuedUtterances})";
                Debug.LogWarning($"LiteRtLmAsrTestRunner continuous: queue full, dropping oldest utterance #{dropped.Index} ({dropped.WavPath})");
                _transcriptLog.Add(dropLine);
            }

            _utteranceQueue.Enqueue(new PendingUtterance
            {
                Index = index,
                WavPath = wavPath,
                Seconds = utteranceSeconds,
                EndpointedAt = Time.realtimeSinceStartup,
            });

            var captureLine = $"[{DateTime.Now:HH:mm:ss}] #{index} captured {utteranceSeconds:0.00}s (queue={_utteranceQueue.Count}/{MaxQueuedUtterances}) -> {wavPath}";
            Debug.Log($"LiteRtLmAsrTestRunner continuous: {captureLine}");
            _transcriptLog.Add(captureLine);
            _logScroll = new Vector2(0f, float.MaxValue);
        }

        private IEnumerator ContinuousWorkerRoutine(AsrModelOption option)
        {
            // Model/tokenizer/VAD assets are resolved once per session and
            // reused for every utterance; combined with the native
            // compiled-model cache this keeps every run after the first warm
            // (compileSeconds ~0 / compiledModelCache=hit in the result
            // JSON, surfaced per entry below).
            string resolvedModelPath = null;
            string resolvedTokenizerPath = null;
            var resolvedSileroPath = string.Empty;
            Exception resolveError = null;
            var vadMode = VadModeOptions[Mathf.Clamp(selectedVadModeIndex, 0, VadModeOptions.Length - 1)];

            var canTranscribe = option != null && _client.IsAvailable;
            if (!canTranscribe)
            {
                Debug.LogWarning("LiteRtLmAsrTestRunner continuous: captures will be saved but not transcribed (no model selected or client unavailable)");
            }
            else
            {
                yield return ResolveStreamingAssetPath(option.modelPath, path => resolvedModelPath = path, ex => resolveError = ex);
                if (resolveError == null)
                {
                    yield return ResolveStreamingAssetPath(option.tokenizerJsonPath, path => resolvedTokenizerPath = path, ex => resolveError = ex);
                }

                if (resolveError == null && IsWhisperGpuRequested(option))
                {
                    var encoderCompanionPath = GetWhisperEncoderCompanionPath(option.modelPath);
                    if (!string.IsNullOrWhiteSpace(encoderCompanionPath))
                    {
                        yield return ResolveStreamingAssetPath(encoderCompanionPath, _ => { }, ex => resolveError = ex);
                    }
                }

                if (resolveError == null && vadMode == "ai")
                {
                    yield return ResolveStreamingAssetPath(SileroVadModelPath, path => resolvedSileroPath = path, ex => resolveError = ex);
                }

                if (resolveError != null)
                {
                    canTranscribe = false;
                    _status = $"Error: {resolveError.Message}";
                    Debug.LogException(resolveError);
                    Debug.LogWarning("LiteRtLmAsrTestRunner continuous: asset resolution failed, captures will be saved but not transcribed");
                }
            }

            while (!_continuousStopRequested)
            {
                if (_utteranceQueue.Count == 0)
                {
                    yield return null;
                    continue;
                }

                var item = _utteranceQueue.Dequeue();
                if (!canTranscribe)
                {
                    _transcriptLog.Add($"[{DateTime.Now:HH:mm:ss}] #{item.Index} saved without ASR: {item.WavPath}");
                    _logScroll = new Vector2(0f, float.MaxValue);
                    continue;
                }

                var asrStartedAt = Time.realtimeSinceStartup;
                var task = System.Threading.Tasks.Task.Run(() =>
                    RunAsrSmokeOnWorkerThread(option, resolvedModelPath, item.WavPath, resolvedTokenizerPath, vadMode, resolvedSileroPath));
                while (!task.IsCompleted)
                {
                    yield return null;
                }

                var asrSeconds = Time.realtimeSinceStartup - asrStartedAt;
                var latencySeconds = Time.realtimeSinceStartup - item.EndpointedAt;
                if (task.IsFaulted || task.IsCanceled)
                {
                    var error = task.Exception?.GetBaseException();
                    var errorLine = $"[{DateTime.Now:HH:mm:ss}] #{item.Index} ASR ERROR: {error?.Message ?? "canceled"}";
                    Debug.LogWarning($"LiteRtLmAsrTestRunner continuous: {errorLine}");
                    _transcriptLog.Add(errorLine);
                    _logScroll = new Vector2(0f, float.MaxValue);
                    continue;
                }

                var asrJson = task.Result;
                var transcript = ExtractTranscript(asrJson);
                var failed = string.IsNullOrWhiteSpace(asrJson) ||
                             asrJson.Contains("\"success\": false", StringComparison.OrdinalIgnoreCase);
                // Keep-warm check: the native compiled-model cache should
                // report hit / compileSeconds ~0 for every run after the
                // session's first.
                var cacheState = ExtractJsonString(asrJson, "compiledModelCache");
                var compileSeconds = ExtractJsonNumber(asrJson, "compileSeconds");
                var cacheLabel = string.IsNullOrEmpty(cacheState)
                    ? (float.IsNaN(compileSeconds) ? "n/a" : $"compile={compileSeconds:0.###}s")
                    : $"{cacheState} (compile={(float.IsNaN(compileSeconds) ? 0f : compileSeconds):0.###}s)";

                _sessionTranscribedCount++;
                _lastUtteranceLatencySeconds = latencySeconds;
                var resultLine =
                    $"[{DateTime.Now:HH:mm:ss}] #{item.Index} ({item.Seconds:0.00}s) " +
                    (failed ? $"FAILED raw={OneLine(Truncate(asrJson, 600))}" : $"transcript={transcript}") +
                    $"\n  latency={latencySeconds:0.00}s (asr={asrSeconds:0.00}s, queueWait={Mathf.Max(0f, latencySeconds - asrSeconds):0.00}s) " +
                    $"modelCache={cacheLabel} queue={_utteranceQueue.Count}/{MaxQueuedUtterances}";
                _transcriptLog.Add(resultLine);
                Debug.Log($"LiteRtLmAsrTestRunner continuous result: {OneLine(resultLine)}");
                _logScroll = new Vector2(0f, float.MaxValue);
                _status = $"Continuous listening... ({_sessionTranscribedCount} transcribed)";
            }

            _continuousActive = false;
            _continuousStopRequested = false;
            _status = "Continuous session ended";
            Debug.Log(
                "LiteRtLmAsrTestRunner continuous: session ended " +
                $"(captured={_sessionUtteranceCount}, transcribed={_sessionTranscribedCount}, dropped={_sessionDroppedCount})");
        }

        // Runs the blocking native ASR call on a Task thread so the main
        // thread (and with it the mic capture loop) never stalls. The JNI
        // thread must be attached before any AndroidJavaObject call and
        // detached afterwards; the bridge object itself was created on the
        // main thread by WarmUpBridge().
        private string RunAsrSmokeOnWorkerThread(
            AsrModelOption option,
            string modelPath,
            string audioPath,
            string tokenizerPath,
            string vadMode,
            string sileroPath)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            AndroidJNI.AttachCurrentThread();
            try
            {
#endif
                return string.Equals(option.mode, QwenAsrMode, StringComparison.OrdinalIgnoreCase)
                    ? _client.RunQwen3AsrSmoke(modelPath, audioPath, tokenizerPath, backend, language, vadMode, sileroPath)
                    : _client.RunWhisperAsrSmoke(modelPath, audioPath, tokenizerPath, backend, language, vadMode, sileroPath);
#if UNITY_ANDROID && !UNITY_EDITOR
            }
            finally
            {
                AndroidJNI.DetachCurrentThread();
            }
#endif
        }

        private static string ExtractJsonString(string json, string key)
        {
            var match = Regex.Match(
                json ?? string.Empty,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"(?<value>(?:\\\\.|[^\"\\\\])*)\"");
            return match.Success ? Regex.Unescape(match.Groups["value"].Value) : string.Empty;
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
            return TranscribeRoutine(option, null, null);
        }

        // overrideAudioPath: absolute path to an already-materialized audio
        // file (e.g. a mic-captured WAV in persistentDataPath); when null the
        // selected bundled audio clip is used.
        private IEnumerator TranscribeRoutine(AsrModelOption option, string overrideAudioPath, string audioLabel)
        {
            _isBusy = true;
            _requestStartedAt = Time.realtimeSinceStartup;
            _status = "Preparing ASR assets...";

            var usesOverrideAudio = !string.IsNullOrWhiteSpace(overrideAudioPath);
            var audioDisplayName = usesOverrideAudio
                ? (audioLabel ?? overrideAudioPath)
                : AudioOptions[selectedAudioIndex];

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
                // ResolveStreamingAssetPath passes rooted paths straight
                // through (with an existence check), so the mic WAV reuses
                // the same resolution flow as the bundled clips.
                var audioSourcePath = usesOverrideAudio ? overrideAudioPath : AudioOptions[selectedAudioIndex];
                yield return ResolveStreamingAssetPath(audioSourcePath, path => resolvedAudioPath = path, ex => resolveError = ex);
            }

            if (resolveError == null && IsWhisperGpuRequested(option))
            {
                var encoderCompanionPath = GetWhisperEncoderCompanionPath(option.modelPath);
                if (!string.IsNullOrWhiteSpace(encoderCompanionPath))
                {
                    yield return ResolveStreamingAssetPath(encoderCompanionPath, _ => { }, ex => resolveError = ex);
                }
            }

            var vadMode = VadModeOptions[Mathf.Clamp(selectedVadModeIndex, 0, VadModeOptions.Length - 1)];
            var resolvedSileroPath = string.Empty;
            if (resolveError == null && vadMode == "ai")
            {
                yield return ResolveStreamingAssetPath(SileroVadModelPath, path => resolvedSileroPath = path, ex => resolveError = ex);
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
                        language,
                        vadMode,
                        resolvedSileroPath)
                    : _client.RunWhisperAsrSmoke(
                        resolvedModelPath,
                        resolvedAudioPath,
                        resolvedTokenizerPath,
                        backend,
                        language,
                        vadMode,
                        resolvedSileroPath);
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
                // Expected-transcript comparison only applies to the bundled
                // clips; mic captures have no reference text.
                var expectedLine = usesOverrideAudio
                    ? string.Empty
                    : DescribeExpectedMatch(selectedAudioIndex, transcript);
                var resultEntry =
                    $"model={option.label}, audio={audioDisplayName}, backend={backend}, language={language}, vadMode={vadMode}\n" +
                    $"elapsedSeconds={elapsedSeconds:0.###}\n" +
                    $"transcript={transcript}\n" +
                    expectedLine +
                    $"raw={OneLine(Truncate(asrJson, 1800))}";
                _transcriptLog.Add(resultEntry);
                Debug.Log($"LiteRtLmAsrTestRunner transcription result: {OneLine(Truncate(resultEntry, 1200))}");
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
