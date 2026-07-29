using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Image + audio input against a multimodal model.
    ///
    /// Rewritten around the shared UI kit. Three things the previous version could
    /// not do and this one can: pick an image from disk, speak into the microphone
    /// instead of only replaying bundled clips, and run on Windows at all — the
    /// desktop path drives `litert_lm_advanced_main` with the `[image:...]` /
    /// `[audio:...]` prompt tags, so selecting audio outside an Android build no
    /// longer silently does nothing.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class LiteRtLmMultimodalTestRunner : MonoBehaviour, ILiteRtLmModelHost
    {
        private enum MediaSource
        {
            None,
            Bundled,
            File,
            Microphone,
        }

        private static readonly string[] ImageSourceLabels = { "None", "Bundled", "File" };
        private static readonly string[] AudioSourceLabels = { "None", "Bundled", "File", "Microphone" };
        private static readonly string[] BackendLabels = { "CPU", "GPU" };

        [SerializeField] private string windowsAdvancedExecutable = "Tools/Windows/Bin/litert_lm_advanced_main.windows_x86_64.exe";
        [SerializeField] private int maxNumTokens = 4000;
        [SerializeField] private string systemInstruction = "You are a helpful multimodal assistant.";

        private LiteRtLmUnityClient _client;
        private LiteRtLmWindowsCliClient _windowsClient;
        private LiteRtLmMicVadCapture _micCapture;

        private int _modelIndex;
        private int _backendIndex;
        private MediaSource _imageSource = MediaSource.Bundled;
        private MediaSource _audioSource = MediaSource.None;
        private int _bundledImageIndex;
        private int _bundledAudioIndex;
        private string _imageFilePath = string.Empty;
        private string _audioFilePath = string.Empty;
        private string _micWavPath;

        private string _prompt = "Describe what you see and hear.";
        private string _response = string.Empty;
        private string _status = "Ready";
        private LiteRtLmUi.StatusKind _statusKind = LiteRtLmUi.StatusKind.Idle;
        private bool _isBusy;
        private bool _isInitialized;

        private readonly LiteRtLmLog _log = new("Multimodal");
        private bool _modelExpanded;
        private bool _backendExpanded;
        private bool _imageExpanded;
        private bool _audioExpanded;
        private Vector2 _logScroll;
        private Vector2 _controlScroll;
        private Vector2 _responseScroll;
        private Texture2D _imagePreview;
        private string _previewSourcePath;

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

            LiteRtLmUi.BeginScreen("Multimodal", out var controlRect, out var outputRect);
            DrawControls(controlRect);
            DrawOutput(outputRect);
        }

        private void DrawControls(Rect rect)
        {
            LiteRtLmUi.BeginPanel(rect, "Multimodal input");
            _controlScroll = GUILayout.BeginScrollView(_controlScroll);

            var width = rect.width - 30f;
            var interactive = !_isBusy;

            _modelIndex = LiteRtLmUi.Dropdown(
                "Model", _modelIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.MultimodalModels),
                ref _modelExpanded, width, interactive);
            _backendIndex = LiteRtLmUi.Dropdown(
                "Backend", _backendIndex, BackendLabels, ref _backendExpanded, width, interactive);

            GUILayout.Label(
                _client.IsAvailable ? "Path: Android bridge"
                : IsDesktop ? "Path: Windows advanced CLI"
                : "Path: unsupported on this platform",
                LiteRtLmUi.Mono);

            // ---- image -------------------------------------------------------
            var imageIndex = LiteRtLmUi.OptionRow("Image", SourceToIndex(_imageSource, false), ImageSourceLabels, width, interactive);
            _imageSource = IndexToSource(imageIndex, false);

            if (_imageSource == MediaSource.Bundled)
            {
                _bundledImageIndex = LiteRtLmUi.Dropdown(
                    null, _bundledImageIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.Images),
                    ref _imageExpanded, width, interactive);
            }
            else if (_imageSource == MediaSource.File)
            {
                _imageFilePath = LiteRtLmUi.PathRow(_imageFilePath, width, interactive, out var browseImage);
                if (browseImage)
                {
                    var picked = OpenFileDialog("Select an image", "jpg,jpeg,png");
                    if (!string.IsNullOrEmpty(picked))
                    {
                        _imageFilePath = picked;
                        Log($"image selected: {Path.GetFileName(picked)}");
                    }
                }
            }

            // ---- audio -------------------------------------------------------
            var audioIndex = LiteRtLmUi.OptionRow("Audio", SourceToIndex(_audioSource, true), AudioSourceLabels, width, interactive);
            _audioSource = IndexToSource(audioIndex, true);

            switch (_audioSource)
            {
                case MediaSource.Bundled:
                    _bundledAudioIndex = LiteRtLmUi.Dropdown(
                        null, _bundledAudioIndex, LiteRtLmSampleAssets.Labels(LiteRtLmSampleAssets.AudioClips),
                        ref _audioExpanded, width, interactive);
                    break;

                case MediaSource.File:
                    _audioFilePath = LiteRtLmUi.PathRow(_audioFilePath, width, interactive, out var browseAudio);
                    if (browseAudio)
                    {
                        var picked = OpenFileDialog("Select audio", "wav,mp3,ogg");
                        if (!string.IsNullOrEmpty(picked))
                        {
                            _audioFilePath = picked;
                            Log($"audio selected: {Path.GetFileName(picked)}");
                        }
                    }

                    break;

                case MediaSource.Microphone:
                    DrawMicSection(interactive);
                    break;
            }

            // ---- prompt ------------------------------------------------------
            LiteRtLmUi.Section("Prompt");
            _prompt = LiteRtLmUi.TextArea(_prompt, 64f);

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUI.enabled = interactive && _client.IsAvailable && !_isInitialized;
            if (GUILayout.Button("Initialize", GUILayout.Height(32f)))
            {
                StartCoroutine(InitializeRoutine());
            }

            GUI.enabled = interactive;
            if (GUILayout.Button("Send", GUILayout.Height(32f)))
            {
                StartCoroutine(SendRoutine());
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
                SetStatus("Stopped", LiteRtLmUi.StatusKind.Idle);
            }

            GUI.enabled = true;
            GUILayout.Label(
                string.IsNullOrEmpty(_micWavPath) ? "(no recording yet)" : $"recorded: {Path.GetFileName(_micWavPath)}",
                LiteRtLmUi.Mono);
        }

        private void DrawOutput(Rect rect)
        {
            LiteRtLmUi.BeginPanel(rect, "Response");

            if (_imageSource != MediaSource.None)
            {
                EnsurePreview();
                if (_imagePreview != null)
                {
                    var previewHeight = Mathf.Min(140f, rect.height * 0.25f);
                    var aspect = _imagePreview.width / (float)Mathf.Max(1, _imagePreview.height);
                    var previewRect = GUILayoutUtility.GetRect(previewHeight * aspect, previewHeight, GUILayout.ExpandWidth(false));
                    GUI.DrawTexture(previewRect, _imagePreview, ScaleMode.ScaleToFit);
                }
            }

            LiteRtLmUi.Section("Model response");
            _responseScroll = LiteRtLmUi.SelectableTextView(
                string.IsNullOrEmpty(_response) ? "(no response yet)" : _response,
                _responseScroll,
                Mathf.Max(80f, rect.height * 0.32f));

            LiteRtLmUi.Section("Log");
            _logScroll = LiteRtLmUi.LogView(_log.Lines, _logScroll, stickToBottom: true);

            LiteRtLmUi.EndPanel();
        }

        private void EnsurePreview()
        {
            var path = ResolveImagePathImmediate();
            if (string.IsNullOrEmpty(path))
            {
                _imagePreview = null;
                _previewSourcePath = null;
                return;
            }

            if (_imagePreview != null && _previewSourcePath == path)
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
                    _imagePreview = texture;
                    _previewSourcePath = path;
                }
            }
            catch (Exception ex)
            {
                Log($"preview failed: {ex.Message}");
            }
        }

        private string ResolveImagePathImmediate()
        {
            return _imageSource switch
            {
                MediaSource.Bundled => LiteRtLmStreamingAssets.Resolve(LiteRtLmSampleAssets.Images[_bundledImageIndex]),
                MediaSource.File => _imageFilePath,
                _ => null,
            };
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
                Fail($"Model not found: {model.ModelPath}. Available: {LiteRtLmStreamingAssets.DescribeAvailable()}");
                yield break;
            }

            try
            {
                var backend = BackendLabels[_backendIndex];
                // Vision and audio executors only load when their backends are set.
                _client.Initialize(modelPath, backend, Application.temporaryCachePath, maxNumTokens,
                    4, 0, false, systemInstruction, backend, "CPU");
                _isInitialized = true;
                SetStatus($"Initialized ({backend}, maxNumTokens={maxNumTokens})", LiteRtLmUi.StatusKind.Idle);
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

        private IEnumerator SendRoutine()
        {
            if (_isBusy)
            {
                yield break;
            }

            _isBusy = true;
            _response = string.Empty;
            SetStatus("Sending...", LiteRtLmUi.StatusKind.Busy);

            // Resolve media first so a missing file is reported before the model runs.
            string imagePath = null;
            if (_imageSource == MediaSource.Bundled)
            {
                yield return ResolveStreamingAsset(LiteRtLmSampleAssets.Images[_bundledImageIndex], p => imagePath = p);
            }
            else if (_imageSource == MediaSource.File)
            {
                imagePath = _imageFilePath;
            }

            string audioPath = null;
            switch (_audioSource)
            {
                case MediaSource.Bundled:
                    yield return ResolveStreamingAsset(LiteRtLmSampleAssets.AudioClips[_bundledAudioIndex], p => audioPath = p);
                    break;
                case MediaSource.File:
                    audioPath = _audioFilePath;
                    break;
                case MediaSource.Microphone:
                    audioPath = _micWavPath;
                    break;
            }

            if (_imageSource != MediaSource.None && string.IsNullOrEmpty(imagePath))
            {
                Fail("No image selected, or the file is missing.");
                yield break;
            }

            if (_audioSource != MediaSource.None && string.IsNullOrEmpty(audioPath))
            {
                Fail(_audioSource == MediaSource.Microphone
                    ? "Record an utterance first."
                    : "No audio selected, or the file is missing.");
                yield break;
            }

            Log($"send: image={Describe(imagePath)} audio={Describe(audioPath)} prompt=\"{LiteRtLmUiText.OneLine(_prompt, 60)}\"");
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
                    _response = _client.SendMessageWithMedia(_prompt, null, imagePath ?? string.Empty, audioPath ?? string.Empty);
                }
                catch (Exception ex)
                {
                    Fail($"Send failed: {ex.Message}");
                    yield break;
                }
            }
            else if (IsDesktop)
            {
                yield return SendViaWindowsCli(imagePath, audioPath);
            }
            else
            {
                Fail("This platform has neither the Android bridge nor the Windows CLI.");
                yield break;
            }

            var elapsed = Time.realtimeSinceStartup - startedAt;
            if (!string.IsNullOrEmpty(_response))
            {
                SetStatus($"Response received in {elapsed:0.0}s", LiteRtLmUi.StatusKind.Idle);
                Log($"response in {elapsed:0.0}s: {LiteRtLmUiText.OneLine(_response, 100)}");
            }
            else
            {
                SetStatus("Empty response", LiteRtLmUi.StatusKind.Error);
                Log("empty response");
            }

            _isBusy = false;
        }

        /// <summary>
        /// Desktop path: `litert_lm_advanced_main` takes media as prompt tags, so
        /// the scene is testable in the editor without an Android build.
        /// </summary>
        private IEnumerator SendViaWindowsCli(string imagePath, string audioPath)
        {
            var model = LiteRtLmSampleAssets.MultimodalModels[_modelIndex];
            var modelPath = LiteRtLmStreamingAssets.Resolve(model.ModelPath);
            if (modelPath == null)
            {
                Fail($"Model not found: {model.ModelPath}");
                yield break;
            }

            var executable = Path.Combine(ProjectRoot(), windowsAdvancedExecutable);
            if (!File.Exists(executable))
            {
                Fail($"Windows CLI not found: {executable}");
                yield break;
            }

            // The [audio:] tag decodes wav only; the bundled clips are mp3, and an
            // unconverted path is passed through as literal text instead of media.
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

                Log($"converted audio to wav: {Path.GetFileName(convertedAudio)}");
                audioPath = convertedAudio;
            }

            var tags = string.Empty;
            if (!string.IsNullOrEmpty(imagePath))
            {
                tags += "[image:" + imagePath.Replace('\\', '/') + "] ";
            }

            if (!string.IsNullOrEmpty(audioPath))
            {
                tags += "[audio:" + audioPath.Replace('\\', '/') + "] ";
            }

            var prompt = tags + _prompt;
            Log($"windows cli: {Path.GetFileName(executable)} {LiteRtLmUiText.OneLine(prompt, 90)}");

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

            _response = task.Result;
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
            _micWavPath = Path.Combine(directory, $"mm_{DateTime.Now:yyyyMMdd_HHmmss_fff}.wav");
            LiteRtLmMicVadCapture.WriteWav16BitMono(_micWavPath, samples, LiteRtLmMicVadCapture.TargetSampleRate);

            var seconds = samples.Length / (float)LiteRtLmMicVadCapture.TargetSampleRate;
            SetStatus($"Recorded {seconds:0.00}s — press Send", LiteRtLmUi.StatusKind.Idle);
            Log($"recorded {seconds:0.00}s from microphone");
        }

        private void HandleMicError(string error)
        {
            SetStatus($"Microphone error: {error}", LiteRtLmUi.StatusKind.Error);
            Log($"mic error: {error}");
        }

        /// <summary>Native file dialog in the editor; typed path elsewhere.</summary>
        private static string OpenFileDialog(string title, string extensions)
        {
#if UNITY_EDITOR
            return UnityEditor.EditorUtility.OpenFilePanel(title, string.Empty, extensions);
#else
            Debug.Log("[LiteRT-LM] File dialogs are editor-only; type the path instead.");
            return null;
#endif
        }

        private static string ProjectRoot() =>
            Path.GetDirectoryName(Application.dataPath) ?? Directory.GetCurrentDirectory();

        private static string Describe(string path) =>
            string.IsNullOrEmpty(path) ? "-" : Path.GetFileName(path);

        private static int SourceToIndex(MediaSource source, bool withMicrophone)
        {
            return source switch
            {
                MediaSource.Bundled => 1,
                MediaSource.File => 2,
                MediaSource.Microphone => withMicrophone ? 3 : 0,
                _ => 0,
            };
        }

        private static MediaSource IndexToSource(int index, bool withMicrophone)
        {
            return index switch
            {
                1 => MediaSource.Bundled,
                2 => MediaSource.File,
                3 when withMicrophone => MediaSource.Microphone,
                _ => MediaSource.None,
            };
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
    }
}
