using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Interactive image (+ optional audio) → tool-call scene.
    ///
    /// The scene used to host the batch demo, which auto-ran one fixed case on
    /// Start and reported through a status file — indistinguishable from a smoke
    /// test. This version lets the tester choose the media, edit the utterance,
    /// the system prompt and the tool list, and run on demand. The batch runner
    /// still exists in the Automated Tests sample for device verification.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class LiteRtLmMultimodalFunctionCallingTestRunner : MonoBehaviour, ILiteRtLmModelHost
    {
        private const string DefaultSystemPrompt =
            "You are a device command router with vision. Look at the media and read the " +
            "utterance, then reply with exactly one JSON object and nothing else: " +
            "{\"tool\":\"<name>\",\"parameters\":{...}}. If no tool fits, reply " +
            "{\"tool\":\"None\",\"parameters\":{}}.";

        // Six mutually exclusive buckets. The router cannot choose without looking
        // at the picture, which is the point. The bundled images cover every branch:
        // apples -> Fruit, puppy-run -> Animal, notebook -> Appliance,
        // princess-snow-white -> Cartoon, couple-together-working-seaside -> Person.
        private const string DefaultToolsJson =
@"[
  {""name"":""HandleFruit"",""description"":""The image mainly shows fruit"",""parameters"":{""fruit"":""string"",""count"":""int""}},
  {""name"":""HandlePerson"",""description"":""The image mainly shows one or more real people (photograph)"",""parameters"":{""count"":""int"",""activity"":""string""}},
  {""name"":""HandleCartoon"",""description"":""The image is an illustration, cartoon or animation still"",""parameters"":{""subject"":""string""}},
  {""name"":""HandleAppliance"",""description"":""The image mainly shows an appliance or electronic device"",""parameters"":{""device"":""string""}},
  {""name"":""HandleAnimal"",""description"":""The image mainly shows an animal"",""parameters"":{""species"":""string"",""count"":""int""}},
  {""name"":""HandleOther"",""description"":""Anything that fits none of the above"",""parameters"":{""summary"":""string""}}
]";

        private static readonly string[] ImageSourceLabels = { "None", "Bundled", "File" };
        private static readonly string[] AudioSourceLabels = { "None", "Bundled", "Microphone" };
        private static readonly string[] BackendLabels = { "CPU", "GPU" };

        [SerializeField] private string windowsExecutable = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
        [SerializeField] private int maxNumTokens = 4000;

        private LiteRtLmUnityClient _client;
        private LiteRtLmWindowsCliClient _windowsClient;
        private LiteRtLmMicVadCapture _micCapture;

        private int _modelIndex;
        private int _backendIndex;
        private int _imageSourceIndex = 1;
        private int _audioSourceIndex;
        private int _bundledImageIndex;
        private int _bundledAudioIndex;
        private string _imageFilePath = string.Empty;
        private string _micWavPath;

        // Neutral on purpose: the tool choice has to come from the image, not
        // from wording in the utterance.
        private string _utterance = "사진을 보고 알맞은 도구를 하나만 호출해줘";
        private string _systemPrompt = DefaultSystemPrompt;
        private string _toolsJson = DefaultToolsJson;

        private string _toolCall = string.Empty;
        private string _rawResponse = string.Empty;
        private string _status = "Ready";
        private LiteRtLmUi.StatusKind _statusKind = LiteRtLmUi.StatusKind.Idle;
        private bool _isBusy;
        private bool _isInitialized;

        private readonly LiteRtLmLog _log = new("MM-FC");
        private bool _modelExpanded;
        private bool _backendExpanded;
        private bool _imageExpanded;
        private bool _audioExpanded;
        private Vector2 _logScroll;
        private Vector2 _controlScroll;
        private Vector2 _outputScroll;
        private Texture2D _preview;
        private string _previewPath;

        private static bool IsDesktop =>
            Application.platform == RuntimePlatform.WindowsEditor ||
            Application.platform == RuntimePlatform.WindowsPlayer;

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
            _windowsClient = null;
            _isInitialized = false;
        }

        private void OnGUI()
        {
            _client ??= new LiteRtLmUnityClient();
            _windowsClient ??= new LiteRtLmWindowsCliClient();

            LiteRtLmUi.BeginScreen("Multimodal → tool call", out var controlRect, out var outputRect);
            DrawControls(controlRect);
            DrawOutput(outputRect);
        }

        private void DrawControls(Rect rect)
        {
            LiteRtLmUi.BeginPanel(rect, "Multimodal → tool call");
            _controlScroll = GUILayout.BeginScrollView(_controlScroll);

            var width = rect.width - 30f;
            var interactive = !_isBusy;

            _modelIndex = LiteRtLmUi.Dropdown(
                "Model", _modelIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.MultimodalModels),
                ref _modelExpanded, width, interactive);
            _backendIndex = LiteRtLmUi.Dropdown(
                "Backend", _backendIndex, BackendLabels, ref _backendExpanded, width, interactive);

            GUILayout.Label(
                _client.IsAvailable ? "Path: Android bridge (real multimodal)"
                : IsDesktop ? "Path: Windows advanced CLI (real vision via [image:] tag)"
                : "Path: unsupported on this platform",
                LiteRtLmUi.Mono);

            _imageSourceIndex = LiteRtLmUi.OptionRow("Image", _imageSourceIndex, ImageSourceLabels, width, interactive);
            if (_imageSourceIndex == 1)
            {
                _bundledImageIndex = LiteRtLmUi.Dropdown(
                    null, _bundledImageIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.Images),
                    ref _imageExpanded, width, interactive);
            }
            else if (_imageSourceIndex == 2)
            {
                _imageFilePath = LiteRtLmUi.PathRow(_imageFilePath, width, interactive, out var browseImage);
                if (browseImage)
                {
#if UNITY_EDITOR
                    var picked = UnityEditor.EditorUtility.OpenFilePanel("Select an image", string.Empty, "jpg,jpeg,png");
                    if (!string.IsNullOrEmpty(picked))
                    {
                        _imageFilePath = picked;
                        Log($"image selected: {Path.GetFileName(picked)}");
                    }
#else
                    Log("File dialogs are editor-only; type the path instead.");
#endif
                }
            }

            _audioSourceIndex = LiteRtLmUi.OptionRow("Audio (optional)", _audioSourceIndex, AudioSourceLabels, width, interactive);
            if (_audioSourceIndex == 1)
            {
                _bundledAudioIndex = LiteRtLmUi.Dropdown(
                    null, _bundledAudioIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.AudioClips),
                    ref _audioExpanded, width, interactive);
            }
            else if (_audioSourceIndex == 2)
            {
                LiteRtLmUi.LevelMeter(
                    LiteRtLmUi.NormalizeDb(_micCapture.CurrentLevelDb),
                    _micCapture.State == LiteRtLmMicVadCapture.MicVadState.Speech,
                    $"{_micCapture.State}  {_micCapture.CurrentLevelDb:0.0} dB");

                GUI.enabled = interactive;
                if (!_micCapture.IsCapturing)
                {
                    if (GUILayout.Button("Record utterance", GUILayout.Height(LiteRtLmUi.RowHeight)))
                    {
                        _micCapture.Continuous = false;
                        _micCapture.StartListening();
                        SetStatus("Listening — speak now", LiteRtLmUi.StatusKind.Busy);
                    }
                }
                else if (GUILayout.Button("Stop", GUILayout.Height(LiteRtLmUi.RowHeight)))
                {
                    _micCapture.StopListening();
                }

                GUI.enabled = true;
            }

            LiteRtLmUi.Section("Utterance");
            _utterance = GUILayout.TextField(_utterance ?? string.Empty, GUILayout.Height(LiteRtLmUi.RowHeight));

            LiteRtLmUi.Section("System prompt");
            _systemPrompt = LiteRtLmUi.TextArea(_systemPrompt, 76f);

            LiteRtLmUi.Section("Tools (JSON)");
            _toolsJson = LiteRtLmUi.TextArea(_toolsJson, 120f);

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUI.enabled = interactive && _client.IsAvailable && !_isInitialized;
            if (GUILayout.Button("Initialize", GUILayout.Height(32f)))
            {
                StartCoroutine(InitializeRoutine());
            }

            GUI.enabled = interactive;
            if (GUILayout.Button("Run", GUILayout.Height(32f)))
            {
                StartCoroutine(RunRoutine());
            }

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

        private void DrawOutput(Rect rect)
        {
            LiteRtLmUi.BeginPanel(rect, "Result");

            EnsurePreview();
            if (_preview != null)
            {
                var previewHeight = Mathf.Min(130f, rect.height * 0.24f);
                var aspect = _preview.width / (float)Mathf.Max(1, _preview.height);
                var previewRect = GUILayoutUtility.GetRect(previewHeight * aspect, previewHeight, GUILayout.ExpandWidth(false));
                GUI.DrawTexture(previewRect, _preview, ScaleMode.ScaleToFit);
            }

            LiteRtLmUi.Section("Tool call");
            GUILayout.TextArea(string.IsNullOrEmpty(_toolCall) ? "(none yet)" : _toolCall, LiteRtLmUi.Mono, GUILayout.Height(64f));

            if (!string.IsNullOrEmpty(_rawResponse) && _rawResponse != _toolCall)
            {
                LiteRtLmUi.Section("Raw response");
                _outputScroll = LiteRtLmUi.SelectableTextView(_rawResponse, _outputScroll, 60f);
            }

            LiteRtLmUi.Section("Log");
            _logScroll = LiteRtLmUi.LogView(_log.Lines, _logScroll, stickToBottom: true);

            LiteRtLmUi.EndPanel();
        }

        private void EnsurePreview()
        {
            var path = _imageSourceIndex switch
            {
                1 => LiteRtLmStreamingAssets.Resolve(LiteRtLmSampleAssets.Images[_bundledImageIndex]),
                2 => _imageFilePath,
                _ => null,
            };

            if (string.IsNullOrEmpty(path))
            {
                _preview = null;
                _previewPath = null;
                return;
            }

            if (_preview != null && _previewPath == path)
            {
                return;
            }

            if (!File.Exists(path))
            {
                return;
            }

            try
            {
                var texture = new Texture2D(2, 2);
                if (texture.LoadImage(File.ReadAllBytes(path)))
                {
                    _preview = texture;
                    _previewPath = path;
                }
            }
            catch (Exception ex)
            {
                Log($"preview failed: {ex.Message}");
            }
        }

        private IEnumerator InitializeRoutine()
        {
            if (_isBusy)
            {
                yield break;
            }

            _isBusy = true;
            SetStatus("Initializing...", LiteRtLmUi.StatusKind.Busy);

            var model = LiteRtLmSampleAssets.MultimodalModels[_modelIndex];
            string modelPath = null;
            yield return ResolveStreamingAsset(model.ModelPath, p => modelPath = p);

            if (modelPath == null)
            {
                Fail($"Model not found: {model.ModelPath}");
                yield break;
            }

            try
            {
                var backend = BackendLabels[_backendIndex];
                var systemInstruction = $"{_systemPrompt}\n\nAvailable tools:\n{_toolsJson}";
                // maxNumTokens must stay high: an image turn plus the tool list
                // overflows a smaller KV cache.
                _client.Initialize(modelPath, backend, Application.temporaryCachePath, maxNumTokens,
                    1, 0, false, systemInstruction, backend, "CPU");
                _isInitialized = true;
                SetStatus($"Initialized ({backend})", LiteRtLmUi.StatusKind.Idle);
                Log($"initialized {model.Label} on {backend}");
            }
            catch (Exception ex)
            {
                Fail($"Initialize failed: {ex.Message}");
                yield break;
            }
            finally
            {
                _isBusy = false;
            }
        }

        private IEnumerator RunRoutine()
        {
            if (_isBusy)
            {
                yield break;
            }

            _isBusy = true;
            _toolCall = string.Empty;
            _rawResponse = string.Empty;
            SetStatus("Running...", LiteRtLmUi.StatusKind.Busy);

            string imagePath = null;
            if (_imageSourceIndex == 1)
            {
                yield return ResolveStreamingAsset(LiteRtLmSampleAssets.Images[_bundledImageIndex], p => imagePath = p);
            }
            else if (_imageSourceIndex == 2)
            {
                imagePath = _imageFilePath;
            }

            string audioPath = null;
            if (_audioSourceIndex == 1)
            {
                yield return ResolveStreamingAsset(LiteRtLmSampleAssets.AudioClips[_bundledAudioIndex], p => audioPath = p);
            }
            else if (_audioSourceIndex == 2)
            {
                audioPath = _micWavPath;
            }

            Log($"run: image={Describe(imagePath)} audio={Describe(audioPath)} utterance=\"{LiteRtLmUiText.OneLine(_utterance, 50)}\"");
            var startedAt = Time.realtimeSinceStartup;

            if (_client.IsAvailable)
            {
                if (!_isInitialized)
                {
                    Fail("Initialize the model first.");
                    yield break;
                }

                try
                {
                    _rawResponse = _client.SendMessageWithMedia(
                        _utterance, null, imagePath ?? string.Empty, audioPath ?? string.Empty);
                }
                catch (Exception ex)
                {
                    Fail($"Run failed: {ex.Message}");
                    yield break;
                }
            }
            else if (IsDesktop)
            {
                yield return RunViaWindowsCli(imagePath, audioPath);
            }
            else
            {
                Fail("This platform has neither the Android bridge nor the Windows CLI.");
                yield break;
            }

            _toolCall = LiteRtLmUiText.ExtractFirstJsonObject(_rawResponse) ?? _rawResponse;
            var elapsed = Time.realtimeSinceStartup - startedAt;
            SetStatus($"Done in {elapsed:0.0}s", LiteRtLmUi.StatusKind.Idle);
            Log($"tool call: {LiteRtLmUiText.OneLine(_toolCall)}");
            _isBusy = false;
        }

        /// <summary>
        /// Desktop path. Uses the advanced CLI so the model genuinely sees the
        /// image: the plain CLI has no vision, and naming the file in the prompt
        /// let the router "classify" from the filename alone — it answered
        /// HandleFruit for apples.jpg without ever looking at the picture.
        /// </summary>
        private IEnumerator RunViaWindowsCli(string imagePath, string audioPath)
        {
            var modelPath = LiteRtLmStreamingAssets.Resolve(LiteRtLmDesktopAsr.DefaultModelRelativePath);
            var executable = Path.Combine(
                Path.GetDirectoryName(Application.dataPath) ?? Directory.GetCurrentDirectory(),
                LiteRtLmDesktopAsr.DefaultExecutableRelativePath);

            if (modelPath == null || !File.Exists(executable))
            {
                Fail(modelPath == null
                    ? $"Multimodal model not found: {LiteRtLmDesktopAsr.DefaultModelRelativePath}"
                    : $"Advanced CLI not found: {executable}");
                yield break;
            }

            // [audio:] decodes wav only.
            if (!string.IsNullOrEmpty(audioPath) &&
                !audioPath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
            {
                string convertedAudio = null;
                string convertError = null;
                yield return LiteRtLmDesktopAsr.EnsureWav(audioPath, p => convertedAudio = p, e => convertError = e);
                if (convertedAudio == null)
                {
                    Fail($"Could not convert audio to wav: {convertError}");
                    yield break;
                }

                audioPath = convertedAudio;
            }

            var tags = string.Empty;
            if (!string.IsNullOrEmpty(imagePath))
            {
                tags += "[image:" + LiteRtLmDesktopAsr.SafeMediaPath(imagePath).Replace("\\", "/") + "] ";
            }

            if (!string.IsNullOrEmpty(audioPath))
            {
                tags += "[audio:" + LiteRtLmDesktopAsr.SafeMediaPath(audioPath).Replace("\\", "/") + "] ";
            }

            var systemInstruction = _systemPrompt + Environment.NewLine + Environment.NewLine +
                                    "Available tools:" + Environment.NewLine + _toolsJson;
            var prompt = tags + _utterance;
            Log($"windows cli (vision): {Path.GetFileName(executable)}");

            var backend = BackendLabels[_backendIndex];
            var task = System.Threading.Tasks.Task.Run(() =>
                _windowsClient.SendMessage(executable, modelPath, prompt, backend, systemInstruction));

            while (!task.IsCompleted)
            {
                yield return null;
            }

            if (task.IsFaulted)
            {
                Fail($"Windows CLI failed: {task.Exception?.GetBaseException().Message}");
                yield break;
            }

            _rawResponse = task.Result;
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

        private void HandleMicUtterance(float[] samples)
        {
            var directory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "MicCaptures");
            Directory.CreateDirectory(directory);
            _micWavPath = Path.Combine(directory, $"mmfc_{DateTime.Now:yyyyMMdd_HHmmss_fff}.wav");
            LiteRtLmMicVadCapture.WriteWav16BitMono(_micWavPath, samples, LiteRtLmMicVadCapture.TargetSampleRate);

            var seconds = samples.Length / (float)LiteRtLmMicVadCapture.TargetSampleRate;
            SetStatus($"Recorded {seconds:0.00}s — press Run", LiteRtLmUi.StatusKind.Idle);
            Log($"recorded {seconds:0.00}s from microphone");
        }

        private void HandleMicError(string error)
        {
            SetStatus($"Microphone error: {error}", LiteRtLmUi.StatusKind.Error);
            Log($"mic error: {error}");
        }

        private static string Describe(string path) =>
            string.IsNullOrEmpty(path) ? "-" : Path.GetFileName(path);

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
    }
}
