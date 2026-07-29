using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Interactive voice → tool-call scene.
    ///
    /// The scene previously hosted the batch demo runner, so it played a fixed clip
    /// through a fixed prompt and printed to the Console. This version exposes the
    /// three things you actually want to vary while testing: the utterance (clip,
    /// microphone, or typed text), the system prompt, and the tool definitions.
    /// The batch runner still exists in the Automated Tests sample for device runs.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class LiteRtLmAsrFunctionCallingTestRunner : MonoBehaviour, ILiteRtLmModelHost
    {
        // The utterance arrives from ASR and is usually Korean, while the tool names
        // and descriptions are English. Without saying so explicitly the router
        // answered "None" for clear matches such as "화면 밝게" -> SetBrightness.
        // Worked examples rather than prose rules: a 0.6B router follows a shown
        // mapping far more reliably than a described one. With the rules alone it
        // answered {"tool":"None"} to plain commands such as "소리 키워 줘".
        private const string DefaultSystemPrompt =
            "You are a device command router. Read the user's utterance and reply with " +
            "exactly one JSON object and nothing else, using double quotes: " +
            "{\"tool\":\"<name>\",\"parameters\":{...}}. " +
            "The utterance may be in Korean while the tools are described in English — " +
            "match on intent, not on wording. " +
            "Prefer a tool over refusing; answer \"None\" only when nothing plausibly fits.\n" +
            "Examples:\n" +
            "소리 키워 줘 -> {\"tool\":\"SetVolume\",\"parameters\":{\"direction\":\"up\",\"amount\":\"70\"}}\n" +
            "볼륨 좀 줄여 -> {\"tool\":\"SetVolume\",\"parameters\":{\"direction\":\"down\",\"amount\":\"30\"}}\n" +
            "화면이 너무 어두워 -> {\"tool\":\"SetBrightness\",\"parameters\":{\"direction\":\"up\"}}\n" +
            "서울 날씨 알려줘 -> {\"tool\":\"GetWeather\",\"parameters\":{\"city\":\"Seoul\"}}\n" +
            "오늘 점심 뭐 먹지 -> {\"tool\":\"None\",\"parameters\":{}}";

        private const string DefaultToolsJson =
@"[
  {""name"":""SetVolume"",""description"":""Change the volume"",""parameters"":{""direction"":""up|down"",""amount"":""int 0-100""}},
  {""name"":""OpenTacticalEvaluationReport"",""description"":""Open a tactical evaluation report"",""parameters"":{""date"":""YYYY-MM-DD""}},
  {""name"":""GetWeather"",""description"":""Get the weather"",""parameters"":{""city"":""string""}},
  {""name"":""SetBrightness"",""description"":""Change screen brightness"",""parameters"":{""direction"":""up|down""}}
]";

        private enum InputSource
        {
            AudioClip,
            Microphone,
            TypedText,
        }

        private static readonly string[] InputSourceLabels = { "Audio clip", "Microphone", "Typed text" };
        private static readonly string[] BackendLabels = { "CPU", "GPU" };
        private static readonly string[] LanguageLabels = { "ko", "en", "auto" };

        [SerializeField] private int llmMaxNumTokens = 1024;

        private LiteRtLmUnityClient _client;
        private LiteRtLmMicVadCapture _micCapture;

        private int _asrModelIndex;
        private int _llmModelIndex;
        private int _asrBackendIndex;
        private int _llmBackendIndex;
        private int _languageIndex;
        private int _clipIndex;
        private InputSource _inputSource = InputSource.AudioClip;

        private string _systemPrompt = DefaultSystemPrompt;
        private string _toolsJson = DefaultToolsJson;
        private string _typedUtterance = "볼륨 올려줘";

        private string _transcript = string.Empty;
        private string _toolCall = string.Empty;
        private string _rawResponse = string.Empty;
        private string _status = "Ready";
        private LiteRtLmUi.StatusKind _statusKind = LiteRtLmUi.StatusKind.Idle;
        private bool _isBusy;

        private readonly LiteRtLmLog _log = new("ASR-FC");
        private bool _asrModelExpanded;
        private bool _llmModelExpanded;
        private bool _clipExpanded;
        private bool _asrBackendExpanded;
        private bool _llmBackendExpanded;
        private bool _languageExpanded;
        private Vector2 _logScroll;
        private Vector2 _controlScroll;
        private Vector2 _outputScroll;

        private void Awake()
        {
            LiteRtLmRunLogOverlay.EnsureCamera();
            _micCapture = gameObject.GetComponent<LiteRtLmMicVadCapture>()
                          ?? gameObject.AddComponent<LiteRtLmMicVadCapture>();
            _micCapture.OnUtteranceCaptured += HandleMicUtterance;
            _micCapture.OnCaptureError += HandleMicError;
        }

        private void OnDestroy() => ReleaseModels();

        /// <inheritdoc />
        public void ReleaseModels()
        {
            if (_micCapture != null)
            {
                _micCapture.OnUtteranceCaptured -= HandleMicUtterance;
                _micCapture.OnCaptureError -= HandleMicError;
                if (_micCapture.IsCapturing)
                {
                    _micCapture.StopListening();
                }
            }

            _client?.Dispose();
            _client = null;
        }

        private void OnGUI()
        {
            _client ??= new LiteRtLmUnityClient();

            LiteRtLmUi.BeginScreen("Voice → tool call", out var controlRect, out var outputRect);

            DrawControls(controlRect);
            DrawOutput(outputRect);
        }

        private void DrawControls(Rect rect)
        {
            LiteRtLmUi.BeginPanel(rect, "Voice → tool call");
            _controlScroll = GUILayout.BeginScrollView(_controlScroll);

            var width = rect.width - 30f;
            var interactive = !_isBusy;

            _asrModelIndex = LiteRtLmUi.Dropdown(
                "ASR model", _asrModelIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.AsrModels),
                ref _asrModelExpanded, width, interactive);
            _llmModelIndex = LiteRtLmUi.Dropdown(
                "LLM router", _llmModelIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.LlmModels),
                ref _llmModelExpanded, width, interactive);

            // Backend and language are dropdowns, matching the ASR scene: they are
            // picked once per session, so they should not occupy a button row each.
            _asrBackendIndex = LiteRtLmUi.Dropdown(
                "ASR backend", _asrBackendIndex, BackendLabels, ref _asrBackendExpanded, width, interactive);
            _llmBackendIndex = LiteRtLmUi.Dropdown(
                "LLM backend", _llmBackendIndex, BackendLabels, ref _llmBackendExpanded, width, interactive);
            _languageIndex = LiteRtLmUi.Dropdown(
                "Language", _languageIndex, LanguageLabels, ref _languageExpanded, width, interactive);

            var sourceIndex = LiteRtLmUi.OptionRow("Utterance source", (int)_inputSource, InputSourceLabels, width, interactive);
            _inputSource = (InputSource)sourceIndex;

            switch (_inputSource)
            {
                case InputSource.AudioClip:
                    _clipIndex = LiteRtLmUi.Dropdown(
                        "Clip", _clipIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.AudioClips),
                        ref _clipExpanded, width, interactive);
                    break;

                case InputSource.Microphone:
                    DrawMicSection(interactive);
                    break;

                case InputSource.TypedText:
                    LiteRtLmUi.Section("Utterance (skips ASR)");
                    _typedUtterance = GUILayout.TextField(_typedUtterance ?? string.Empty, GUILayout.Height(LiteRtLmUi.RowHeight));
                    break;
            }

            LiteRtLmUi.Section("System prompt");
            _systemPrompt = LiteRtLmUi.TextArea(_systemPrompt, 84f);

            LiteRtLmUi.Section("Tools (JSON)");
            _toolsJson = LiteRtLmUi.TextArea(_toolsJson, 130f);

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUI.enabled = interactive && _inputSource != InputSource.Microphone;
            if (GUILayout.Button("Run", GUILayout.Height(32f)))
            {
                StartCoroutine(RunRoutine(null));
            }

            GUI.enabled = interactive;
            if (GUILayout.Button("Reset prompt", GUILayout.Width(110f), GUILayout.Height(32f)))
            {
                _systemPrompt = DefaultSystemPrompt;
                _toolsJson = DefaultToolsJson;
            }

            GUI.enabled = true;
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            LiteRtLmUi.Status(_status, _statusKind);

            GUILayout.EndScrollView();
            LiteRtLmUi.EndPanel();
        }

        private void DrawMicSection(bool interactive)
        {
            LiteRtLmUi.Section("Microphone");
            LiteRtLmUi.LevelMeter(
                LiteRtLmUi.NormalizeDb(_micCapture.CurrentLevelDb),
                _micCapture.State == LiteRtLmMicVadCapture.MicVadState.Speech,
                $"{_micCapture.State}  {_micCapture.CurrentLevelDb:0.0} dB");

            GUI.enabled = interactive;
            if (!_micCapture.IsCapturing)
            {
                if (GUILayout.Button("Listen once", GUILayout.Height(LiteRtLmUi.RowHeight)))
                {
                    _micCapture.Continuous = false;
                    _micCapture.StartListening();
                    SetStatus("Listening — speak now", LiteRtLmUi.StatusKind.Busy);
                }
            }
            else if (GUILayout.Button("Stop listening", GUILayout.Height(LiteRtLmUi.RowHeight)))
            {
                _micCapture.StopListening();
                SetStatus("Stopped", LiteRtLmUi.StatusKind.Idle);
            }

            GUI.enabled = true;
        }

        private void DrawOutput(Rect rect)
        {
            LiteRtLmUi.BeginPanel(rect, "Result");

            LiteRtLmUi.Section("Transcript");
            _outputScroll = LiteRtLmUi.SelectableTextView(_transcript, _outputScroll, 60f);

            LiteRtLmUi.Section("Tool call");
            GUILayout.TextArea(string.IsNullOrEmpty(_toolCall) ? "(none yet)" : _toolCall, LiteRtLmUi.Mono, GUILayout.Height(60f));

            if (!string.IsNullOrEmpty(_rawResponse) && _rawResponse != _toolCall)
            {
                LiteRtLmUi.Section("Raw LLM response");
                GUILayout.TextArea(_rawResponse, LiteRtLmUi.Mono, GUILayout.Height(52f));
            }

            LiteRtLmUi.Section("Log");
            _logScroll = LiteRtLmUi.LogView(_log.Lines, _logScroll, stickToBottom: true);

            LiteRtLmUi.EndPanel();
        }

        private void HandleMicUtterance(float[] samples)
        {
            var directory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "MicCaptures");
            Directory.CreateDirectory(directory);
            var wavPath = Path.Combine(directory, $"fc_{DateTime.Now:yyyyMMdd_HHmmss_fff}.wav");
            LiteRtLmMicVadCapture.WriteWav16BitMono(wavPath, samples, LiteRtLmMicVadCapture.TargetSampleRate);

            Log($"captured {samples.Length / (float)LiteRtLmMicVadCapture.TargetSampleRate:0.00}s from microphone");
            StartCoroutine(RunRoutine(wavPath));
        }

        private void HandleMicError(string error)
        {
            SetStatus($"Microphone error: {error}", LiteRtLmUi.StatusKind.Error);
            Log($"mic error: {error}");
        }

        /// <summary>
        /// Runs the pipeline. <paramref name="overrideAudioPath"/> is an absolute
        /// path (microphone capture); otherwise the selected clip or typed text is used.
        /// </summary>
        private IEnumerator RunRoutine(string overrideAudioPath)
        {
            if (_isBusy)
            {
                yield break;
            }

            _isBusy = true;
            _transcript = string.Empty;
            _toolCall = string.Empty;
            _rawResponse = string.Empty;

            var asrModel = LiteRtLmSampleAssets.AsrModels[_asrModelIndex];
            var llmModel = LiteRtLmSampleAssets.LlmModels[_llmModelIndex];
            var language = LanguageLabels[_languageIndex];

            string utterance;

            if (_inputSource == InputSource.TypedText && overrideAudioPath == null)
            {
                utterance = _typedUtterance;
                _transcript = utterance;
                Log($"typed utterance: {utterance}");
            }
            else
            {
                SetStatus("Transcribing...", LiteRtLmUi.StatusKind.Busy);

                string audioPath = overrideAudioPath;
                if (audioPath == null)
                {
                    var clip = LiteRtLmSampleAssets.AudioClips[_clipIndex];
                    yield return ResolveStreamingAsset(clip, p => audioPath = p);
                }

                string modelPath = null, tokenizerPath = null;
                yield return ResolveStreamingAsset(asrModel.ModelPath, p => modelPath = p);
                yield return ResolveStreamingAsset(asrModel.TokenizerPath, p => tokenizerPath = p);

                if (audioPath == null || (_client.IsAvailable && (modelPath == null || tokenizerPath == null)))
                {
                    Fail($"ASR assets missing (model={modelPath != null}, tokenizer={tokenizerPath != null}, audio={audioPath != null})");
                    yield break;
                }

                string asrJson = null;

                if (!_client.IsAvailable)
                {
                    // Editor: the whisper tflite path is Android-only, so transcribe
                    // through the desktop CLI instead of failing outright.
                    if (!LiteRtLmDesktopAsr.IsSupported)
                    {
                        Fail("ASR needs the Android bridge or the Windows CLI.");
                        yield break;
                    }

                    LiteRtLmDesktopAsr.Result desktop = null;
                    yield return LiteRtLmDesktopAsr.Transcribe(
                        audioPath, language, BackendLabels[_asrBackendIndex], r => desktop = r);

                    if (desktop == null || !desktop.Success)
                    {
                        Fail($"ASR failed: {desktop?.Error ?? "unknown failure"}");
                        yield break;
                    }

                    Log($"desktop ASR {desktop.ElapsedSeconds:0.0}s");
                    utterance = desktop.Transcript;
                }
                else
                {
                    try
                    {
                        asrJson = asrModel.Mode == "qwen3"
                            ? _client.RunQwen3AsrSmoke(modelPath, audioPath, tokenizerPath, BackendLabels[_asrBackendIndex], language, "energy", null)
                            : _client.RunWhisperAsrSmoke(modelPath, audioPath, tokenizerPath, BackendLabels[_asrBackendIndex], language, "energy", null);
                    }
                    catch (Exception ex)
                    {
                        Fail($"ASR failed: {ex.Message}");
                        yield break;
                    }

                    utterance = ExtractJsonString(asrJson, "transcript");
                }

                _transcript = utterance;
                Log($"transcript: {utterance}");

                if (string.IsNullOrWhiteSpace(utterance))
                {
                    Fail("ASR produced an empty transcript");
                    yield break;
                }
            }

            SetStatus("Routing to a tool...", LiteRtLmUi.StatusKind.Busy);

            string llmPath = null;
            yield return ResolveStreamingAsset(llmModel.ModelPath, p => llmPath = p);
            if (llmPath == null)
            {
                Fail($"LLM model not found: {llmModel.ModelPath}");
                yield break;
            }

            var systemInstruction = _systemPrompt + Environment.NewLine + Environment.NewLine +
                                    "Available tools:" + Environment.NewLine + _toolsJson;

            // No Android bridge in the editor: route through the Windows CLI so the
            // scene completes instead of stopping right after transcription.
            if (!_client.IsAvailable)
            {
                yield return RouteViaWindowsCli(llmPath, systemInstruction, utterance);
                yield break;
            }

            try
            {
                _client.Dispose();
                _client = new LiteRtLmUnityClient();

                _client.Initialize(llmPath, BackendLabels[_llmBackendIndex], Application.temporaryCachePath,
                    llmMaxNumTokens, 0, 0, false, systemInstruction);
                _client.ResetConversation(systemInstruction);

                _rawResponse = _client.SendMessage($"Utterance: {utterance}");
                _toolCall = ExtractFirstJsonObject(_rawResponse) ?? _rawResponse;
                Log($"tool call: {LiteRtLmUiText.OneLine(_toolCall)}");
                SetStatus("Done", LiteRtLmUi.StatusKind.Idle);
            }
            catch (Exception ex)
            {
                Fail($"LLM failed: {ex.Message}");
                yield break;
            }
            finally
            {
                _isBusy = false;
            }
        }

        /// <summary>Desktop routing step: the same tool contract through the Windows CLI.</summary>
        private IEnumerator RouteViaWindowsCli(string modelPath, string systemInstruction, string utterance)
        {
            var executable = Path.Combine(
                Path.GetDirectoryName(Application.dataPath) ?? Directory.GetCurrentDirectory(),
                "Tools/Windows/Bin/litert_lm_main.windows_x86_64.exe");

            if (!File.Exists(executable))
            {
                Fail($"Windows CLI not found: {executable}");
                yield break;
            }

            var cli = new LiteRtLmWindowsCliClient();
            var prompt = "Utterance: " + utterance;
            var backend = BackendLabels[_llmBackendIndex];
            var task = System.Threading.Tasks.Task.Run(() =>
                cli.SendMessage(executable, modelPath, prompt, backend, systemInstruction));

            while (!task.IsCompleted)
            {
                yield return null;
            }

            if (task.IsFaulted)
            {
                Fail($"LLM failed: {task.Exception?.GetBaseException().Message}");
                yield break;
            }

            _rawResponse = task.Result ?? string.Empty;
            _toolCall = ExtractFirstJsonObject(_rawResponse) ?? _rawResponse;
            Log($"tool call: {LiteRtLmUiText.OneLine(_toolCall)}");
            SetStatus("Done (Windows CLI)", LiteRtLmUi.StatusKind.Idle);
            _isBusy = false;
        }

        private IEnumerator ResolveStreamingAsset(string relativePath, Action<string> onResolved)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            var source = $"{Application.streamingAssetsPath}/{relativePath}";
            var destination = Path.Combine(Application.persistentDataPath, "LiteRTLM", relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            if (!File.Exists(destination))
            {
                using var request = UnityWebRequest.Get(source);
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Log($"failed to stage '{relativePath}': {request.error}");
                    onResolved(null);
                    yield break;
                }

                File.WriteAllBytes(destination, request.downloadHandler.data);
            }

            onResolved(destination);
#else
            onResolved(LiteRtLmStreamingAssets.Resolve(relativePath));
            yield break;
#endif
        }

        private void SetStatus(string status, LiteRtLmUi.StatusKind kind)
        {
            _status = status;
            _statusKind = kind;
        }

        private void Fail(string message)
        {
            SetStatus(message, LiteRtLmUi.StatusKind.Error);
            _log.Error(message);
            _isBusy = false;
        }

        private void Log(string line) => _log.Info(line);

        private static string ExtractJsonString(string json, string key) =>
            LiteRtLmUiText.ExtractJsonString(json, key);

        private static string ExtractFirstJsonObject(string text) =>
            LiteRtLmUiText.ExtractFirstJsonObject(text);
    }
}
