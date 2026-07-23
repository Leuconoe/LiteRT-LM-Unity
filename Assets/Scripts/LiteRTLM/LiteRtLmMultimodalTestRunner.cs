using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmMultimodalTestRunner : MonoBehaviour
    {
        // UnityLiteRtLmBridge exposes sendMessageWithMedia since the take3 AAR
        // (text, byte[] imageBytes, imagePath, audioPath, extraContextJson).
        private const bool MediaApiAvailable = true;

        private static readonly string[] AudioOptions =
        {
            "(none)",
            "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3",
            "TestAssets/Audio/Tactical Evaluation Results Report - March 5, 2025.mp3",
        };

        [Header("LiteRT-LM Multimodal Test")]
        [SerializeField] private string modelPath = "Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm";
        [SerializeField] private string backend = "GPU";
        [SerializeField] private int maxNumTokens = 1024;
        [SerializeField] private int maxNumImages = 4;
        // Engine loads vision/audio executors only when these are non-empty.
        [SerializeField] private string visionBackend = "CPU";
        [SerializeField] private string audioBackend = "";
        [SerializeField] private string systemInstruction = "You are a helpful multimodal assistant.";
        [SerializeField] private Texture2D[] testImages = Array.Empty<Texture2D>();

        private LiteRtLmUnityClient _client;
        private string _status = "Idle";
        private string _response = "";
        private string _prompt = "Describe the selected media.";
        private string _resolvedModelPath = "";
        private int _selectedImageIndex = -1;
        private int _selectedAudioIndex;
        private bool _audioListExpanded;
        private bool _isBusy;
        private bool _isInitialized;
        private Vector2 _responseScroll;
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
            GUILayout.Label("LiteRT-LM Multimodal Test");
            GUILayout.Label($"Status: {_status}");

            var canUseAndroidBridge = _client.IsAvailable;
            if (!canUseAndroidBridge)
            {
                GUILayout.Label("Multimodal inference is unsupported on this platform (Android device builds only).");
            }

            _hasImeTextFieldFocus = false;
            GUILayout.Label("Model Path (StreamingAssets-relative file name)");
            modelPath = DrawImeTextField(modelPath, "LiteRtLmMultimodalModelField");
            GUILayout.Label("Backend");
            backend = DrawImeTextField(backend, "LiteRtLmMultimodalBackendField");

            DrawImagePickerRow();
            DrawAudioDropdown();

            GUILayout.Label("Prompt");
            _prompt = DrawImeTextField(_prompt, "LiteRtLmMultimodalPromptField");
            if (!_hasImeTextFieldFocus)
            {
                Input.imeCompositionMode = IMECompositionMode.Auto;
            }

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Initialize", GUILayout.Height(40)) && !_isBusy)
            {
                if (canUseAndroidBridge)
                {
                    StartCoroutine(InitializeRoutine());
                }
                else
                {
                    _status = "Initialize requires an Android device build.";
                }
            }

            if (GUILayout.Button("Send", GUILayout.Height(40)) && !_isBusy)
            {
                Send(canUseAndroidBridge);
            }

            GUILayout.EndHorizontal();

            GUILayout.Space(8);
            GUILayout.Label("Response");
            _responseScroll = GUILayout.BeginScrollView(_responseScroll, GUILayout.Height(320));
            GUILayout.TextArea(_response, GUILayout.ExpandHeight(true));
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private void DrawImagePickerRow()
        {
            GUILayout.Label("Test Image");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(_selectedImageIndex < 0 ? "✓ None" : "None", GUILayout.Width(90), GUILayout.Height(72)))
            {
                _selectedImageIndex = -1;
            }

            if (testImages != null)
            {
                for (var i = 0; i < testImages.Length; i++)
                {
                    var image = testImages[i];
                    if (image == null)
                    {
                        continue;
                    }

                    var selectedMark = _selectedImageIndex == i ? "✓ " : string.Empty;
                    if (GUILayout.Button(new GUIContent(image, selectedMark + image.name), GUILayout.Width(90), GUILayout.Height(72)))
                    {
                        _selectedImageIndex = i;
                    }
                }
            }

            GUILayout.EndHorizontal();
            GUILayout.Label(_selectedImageIndex < 0 || testImages == null || _selectedImageIndex >= testImages.Length || testImages[_selectedImageIndex] == null
                ? "Selected image: none"
                : $"Selected image: {testImages[_selectedImageIndex].name}");
        }

        private void DrawAudioDropdown()
        {
            GUILayout.Label("Audio (StreamingAssets)");
            _selectedAudioIndex = Mathf.Clamp(_selectedAudioIndex, 0, AudioOptions.Length - 1);
            if (GUILayout.Button($"{AudioOptions[_selectedAudioIndex]} {(_audioListExpanded ? "▲" : "▼")}"))
            {
                _audioListExpanded = !_audioListExpanded;
            }

            if (!_audioListExpanded)
            {
                return;
            }

            for (var i = 0; i < AudioOptions.Length; i++)
            {
                if (GUILayout.Button((i == _selectedAudioIndex ? "✓ " : "   ") + AudioOptions[i]))
                {
                    _selectedAudioIndex = i;
                    _audioListExpanded = false;
                }
            }
        }

        private void Send(bool canUseAndroidBridge)
        {
            if (!canUseAndroidBridge)
            {
                _status = "Send is unsupported on this platform (Android device builds only).";
                return;
            }

            if (!_isInitialized)
            {
                _status = "Initialize first.";
                return;
            }

            var hasImage = _selectedImageIndex >= 0 &&
                           testImages != null &&
                           _selectedImageIndex < testImages.Length &&
                           testImages[_selectedImageIndex] != null;
            var hasAudio = _selectedAudioIndex > 0;
            if ((hasImage || hasAudio) && !MediaApiAvailable)
            {
                _status = "bridge API pending: SendMessageWithMedia";
                _response = "The Android bridge does not expose SendMessageWithMedia yet. " +
                            "Text-only prompts work today; media attachments require a bridge update.";
                return;
            }

            StartCoroutine(SendRoutine(hasImage, hasAudio));
        }

        private IEnumerator SendRoutine(bool hasImage, bool hasAudio)
        {
            _isBusy = true;
            _status = "Generating response...";

            byte[] imageBytes = null;
            if (hasImage)
            {
                try
                {
                    imageBytes = testImages[_selectedImageIndex].EncodeToPNG();
                }
                catch (Exception ex)
                {
                    _status = $"Error: test image is not readable ({ex.Message}). Enable Read/Write on the texture import settings.";
                    Debug.LogException(ex);
                    _isBusy = false;
                    yield break;
                }
            }

            var resolvedAudioPath = string.Empty;
            if (hasAudio)
            {
                Exception audioError = null;
                yield return ResolveStreamingAssetPath(
                    AudioOptions[_selectedAudioIndex],
                    path => resolvedAudioPath = path,
                    ex => audioError = ex);
                if (audioError != null)
                {
                    _status = $"Error: {audioError.Message}";
                    Debug.LogException(audioError);
                    _isBusy = false;
                    yield break;
                }
            }

            try
            {
                _response = imageBytes != null || hasAudio
                    ? _client.SendMessageWithMedia(_prompt, imageBytes, string.Empty, resolvedAudioPath)
                    : _client.SendMessage(_prompt);
                _status = "Response received";
            }
            catch (Exception ex)
            {
                _status = $"Error: {ex.Message}";
                Debug.LogException(ex);
            }
            finally
            {
                _isBusy = false;
            }
        }

        private IEnumerator InitializeRoutine()
        {
            _isBusy = true;
            _status = "Preparing model...";

            string resolvedPath = null;
            Exception error = null;
            yield return ResolveStreamingAssetPath(modelPath, path => resolvedPath = path, ex => error = ex);

            if (error != null)
            {
                _status = $"Error: {error.Message}";
                Debug.LogException(error);
                _isBusy = false;
                yield break;
            }

            try
            {
                _resolvedModelPath = resolvedPath;
                _client.Dispose();
                _client = new LiteRtLmUnityClient();
                _client.Initialize(
                    _resolvedModelPath,
                    backend: backend,
                    maxNumTokens: maxNumTokens,
                    maxNumImages: maxNumImages,
                    systemInstruction: systemInstruction,
                    visionBackend: visionBackend,
                    audioBackend: audioBackend);
                _isInitialized = true;
                _status = $"Initialized (maxNumImages={maxNumImages})";
            }
            catch (Exception ex)
            {
                _isInitialized = false;
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
                onError(new ArgumentException("Model path or StreamingAssets file name is required."));
                yield break;
            }

            if (Path.IsPathRooted(configuredPath))
            {
                onSuccess(configuredPath);
                yield break;
            }

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM", "Multimodal");
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
            if (!sourcePath.Contains("://"))
            {
                if (!File.Exists(sourcePath))
                {
                    onError(new FileNotFoundException($"File not found in StreamingAssets: {sourcePath}"));
                    yield break;
                }

                onSuccess(sourcePath);
                yield break;
            }

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

            onSuccess(destinationPath);
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
    }
}
