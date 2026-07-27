using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmLlmChatTestRunner : MonoBehaviour, ILiteRtLmModelHost
    {
        private static readonly string[] ModelOptions =
        {
            "Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm",
            "LLM/gemma3-1b/gemma3-1b-it-int4.litertlm",
            "LLM/gemma3-270m/gemma3-270m-it-q8.litertlm",
            "LLM/qwen3-0.6b/Qwen3-0.6B.litertlm",
            "LLM/qwen2.5-0.5b/Qwen2.5-0.5B-Instruct-q8.litertlm",
        };

        [Header("LiteRT-LM LLM Chat Test")]
        [SerializeField] private int selectedModelIndex;
        [SerializeField] private string backend = "GPU";
        [SerializeField] private string systemInstruction = "You are a helpful assistant.";
        [SerializeField] private int maxNumTokens = 1024;
        [SerializeField] private string windowsCliExecutablePath = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
        [SerializeField] private string windowsBackend = "GPU";
        [SerializeField] private float windowsRequestTimeoutSeconds = 120f;
        [SerializeField] private bool enableThinking;

        private LiteRtLmUnityClient _client;
        private LiteRtLmWindowsCliClient _windowsCliClient;
        private readonly List<ChatTurn> _transcript = new List<ChatTurn>();
        private string _status = "Idle";
        private string _pendingInput = "";
        private string _resolvedModelPath = "";
        private string _resolvedWindowsExecutablePath = "";
        private bool _isBusy;
        private bool _isInitialized;
        private bool _modelListExpanded;
        private float _requestStartedAt;
        private float _lastElapsedSeconds;
        private Vector2 _transcriptScroll;
        private bool _hasImeTextFieldFocus;
        private CancellationTokenSource _windowsRequestCancellationTokenSource;

        private void Awake()
        {
            EnsureClients();
        }

        private void OnDestroy()
        {
            ReleaseModels();
        }

        /// <inheritdoc />
        public void ReleaseModels()
        {
            _windowsRequestCancellationTokenSource?.Cancel();
            _windowsRequestCancellationTokenSource?.Dispose();
            _windowsRequestCancellationTokenSource = null;
            _client?.Dispose();
            _client = null;
            _windowsCliClient = null;
        }

        private void OnGUI()
        {
            EnsureClients();

            GUILayout.BeginArea(new Rect(20, 20, 760, 860), GUI.skin.box);
            GUILayout.Label("LiteRT-LM LLM Chat Test");

            var canUseAndroidBridge = _client.IsAvailable;
            var canUseWindowsCli = Application.platform == RuntimePlatform.WindowsEditor ||
                                   Application.platform == RuntimePlatform.WindowsPlayer;

            if (!canUseAndroidBridge && !canUseWindowsCli)
            {
                GUILayout.Label("This test supports Android device builds and Windows CLI testing.");
                GUILayout.EndArea();
                return;
            }

            GUILayout.Label($"Status: {_status}");
            GUILayout.Label(canUseAndroidBridge ? "Mode: Android bridge" : "Mode: Windows CLI fallback");
            if (_isBusy)
            {
                GUILayout.Label($"Elapsed: {Time.realtimeSinceStartup - _requestStartedAt:0.0}s");
            }
            else if (_lastElapsedSeconds > 0f)
            {
                GUILayout.Label($"Last response time: {_lastElapsedSeconds:0.00}s");
            }

            _hasImeTextFieldFocus = false;
            DrawModelDropdown();
            GUILayout.Label("Backend");
            backend = DrawImeTextField(backend, "LiteRtLmChatBackendField");
            if (CurrentModelSupportsThinking())
            {
                enableThinking = GUILayout.Toggle(enableThinking, "Thinking (append /think soft switch)");
            }
            if (canUseWindowsCli && !canUseAndroidBridge)
            {
                GUILayout.Label("Windows CLI Executable Path");
                windowsCliExecutablePath = DrawImeTextField(windowsCliExecutablePath, "LiteRtLmChatWindowsExeField");
                GUILayout.Label("Windows Backend");
                windowsBackend = DrawImeTextField(windowsBackend, "LiteRtLmChatWindowsBackendField");
            }

            if (GUILayout.Button("Initialize", GUILayout.Height(36)) && !_isBusy)
            {
                StartCoroutine(InitializeRoutine(canUseAndroidBridge));
            }

            GUILayout.Space(8);
            GUILayout.Label("Transcript");
            _transcriptScroll = GUILayout.BeginScrollView(_transcriptScroll, GUILayout.Height(420));
            for (var i = 0; i < _transcript.Count; i++)
            {
                var turn = _transcript[i];
                GUILayout.Label($"[{turn.Role}]");
                GUILayout.TextArea(turn.Text);
                if (!string.IsNullOrEmpty(turn.Thoughts))
                {
                    if (GUILayout.Button($"[thoughts] {(turn.ShowThoughts ? "▲" : "▼")}", GUILayout.Width(120)))
                    {
                        turn.ShowThoughts = !turn.ShowThoughts;
                        _transcript[i] = turn;
                    }

                    if (turn.ShowThoughts)
                    {
                        GUILayout.TextArea(turn.Thoughts);
                    }
                }

                GUILayout.Space(4);
            }

            GUILayout.EndScrollView();

            GUILayout.Label("Message");
            _pendingInput = DrawImeTextField(_pendingInput, "LiteRtLmChatInputField");
            if (!_hasImeTextFieldFocus)
            {
                Input.imeCompositionMode = IMECompositionMode.Auto;
            }

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Send", GUILayout.Height(36)) && !_isBusy)
            {
                SendPendingInput(canUseAndroidBridge);
            }

            if (GUILayout.Button("Reset", GUILayout.Height(36)) && !_isBusy)
            {
                ResetConversation(canUseAndroidBridge);
            }

            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void DrawModelDropdown()
        {
            GUILayout.Label("Model");
            selectedModelIndex = Mathf.Clamp(selectedModelIndex, 0, ModelOptions.Length - 1);
            if (GUILayout.Button($"{ModelOptions[selectedModelIndex]} {(_modelListExpanded ? "▲" : "▼")}"))
            {
                _modelListExpanded = !_modelListExpanded;
            }

            if (!_modelListExpanded)
            {
                return;
            }

            for (var i = 0; i < ModelOptions.Length; i++)
            {
                if (GUILayout.Button((i == selectedModelIndex ? "✓ " : "   ") + ModelOptions[i]))
                {
                    if (i != selectedModelIndex)
                    {
                        selectedModelIndex = i;
                        _isInitialized = false;
                        _status = "Model changed. Initialize again.";
                    }

                    _modelListExpanded = false;
                }
            }
        }

        private void SendPendingInput(bool canUseAndroidBridge)
        {
            var text = (_pendingInput ?? string.Empty).Trim();
            if (string.IsNullOrEmpty(text))
            {
                _status = "Type a message first.";
                return;
            }

            if (!_isInitialized)
            {
                _status = "Initialize first.";
                return;
            }

            var outgoingText = text;
            if (CurrentModelSupportsThinking())
            {
                outgoingText = text + (enableThinking ? " /think" : " /no_think");
            }

            if (canUseAndroidBridge)
            {
                try
                {
                    _isBusy = true;
                    _requestStartedAt = Time.realtimeSinceStartup;
                    _status = "Generating response...";
                    var response = _client.SendMessage(outgoingText);
                    _lastElapsedSeconds = Time.realtimeSinceStartup - _requestStartedAt;
                    var answer = ExtractThinkBlocks(response, out var thoughts);
                    AppendTurn("user", outgoingText);
                    AppendTurn("assistant", answer, thoughts);
                    _pendingInput = "";
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
            else
            {
                StartCoroutine(SendWindowsCliMessageRoutine(outgoingText));
            }
        }

        private void ResetConversation(bool canUseAndroidBridge)
        {
            _transcript.Clear();
            _pendingInput = "";
            _lastElapsedSeconds = 0f;
            if (canUseAndroidBridge && _isInitialized)
            {
                try
                {
                    _client.ResetConversation(systemInstruction);
                    _status = "Conversation reset";
                }
                catch (Exception ex)
                {
                    _status = $"Error: {ex.Message}";
                    Debug.LogException(ex);
                }
            }
            else
            {
                _status = "Conversation reset";
            }
        }

        private IEnumerator InitializeRoutine(bool canUseAndroidBridge)
        {
            _isBusy = true;
            _status = "Preparing model...";

            string resolvedPath = null;
            Exception error = null;
            yield return ResolveModelPathCoroutine(
                ModelOptions[Mathf.Clamp(selectedModelIndex, 0, ModelOptions.Length - 1)],
                path => resolvedPath = path,
                ex => error = ex);

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
                if (canUseAndroidBridge)
                {
                    _client.Dispose();
                    _client = new LiteRtLmUnityClient();
                    _client.Initialize(
                        _resolvedModelPath,
                        backend: backend,
                        maxNumTokens: maxNumTokens,
                        systemInstruction: systemInstruction);
                    _status = "Initialized (Android native session)";
                }
                else
                {
                    _resolvedWindowsExecutablePath = ResolveWindowsExecutablePath(windowsCliExecutablePath);
                    if (!File.Exists(_resolvedWindowsExecutablePath))
                    {
                        throw new FileNotFoundException(
                            $"Windows executable not found: {_resolvedWindowsExecutablePath}",
                            _resolvedWindowsExecutablePath);
                    }

                    _status = "Initialized (Windows CLI, history via messagesJson)";
                }

                _transcript.Clear();
                _isInitialized = true;
            }
            catch (Exception ex)
            {
                _isInitialized = false;
                _status = $"Error: {ex.Message}";
                Debug.LogException(ex);
            }

            _isBusy = false;
        }

        private IEnumerator SendWindowsCliMessageRoutine(string text)
        {
            _isBusy = true;
            _requestStartedAt = Time.realtimeSinceStartup;
            _status = "Running Windows CLI...";

            _windowsRequestCancellationTokenSource?.Dispose();
            _windowsRequestCancellationTokenSource = new CancellationTokenSource();
            var task = _windowsCliClient.SendMessageAsync(
                _resolvedWindowsExecutablePath,
                _resolvedModelPath,
                text,
                windowsBackend,
                TimeSpan.FromSeconds(Mathf.Max(1f, windowsRequestTimeoutSeconds)),
                _windowsRequestCancellationTokenSource.Token,
                systemMessage: systemInstruction,
                messagesJson: BuildMessagesJson());

            while (!task.IsCompleted)
            {
                yield return null;
            }

            _lastElapsedSeconds = Time.realtimeSinceStartup - _requestStartedAt;
            if (task.IsFaulted)
            {
                var ex = task.Exception?.GetBaseException() ?? new InvalidOperationException("Windows CLI task failed.");
                _status = $"Error: {ex.Message}";
                Debug.LogException(ex);
            }
            else if (task.IsCanceled)
            {
                _status = "Windows CLI request canceled.";
            }
            else
            {
                var answer = ExtractThinkBlocks(StripBenchmarkInfo(task.Result), out var thoughts);
                AppendTurn("user", text);
                AppendTurn("assistant", answer, thoughts);
                _pendingInput = "";
                _status = "Response received";
            }

            _isBusy = false;
        }

        private string BuildMessagesJson()
        {
            if (_transcript.Count == 0)
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            builder.Append('[');
            for (var i = 0; i < _transcript.Count; i++)
            {
                if (i > 0)
                {
                    builder.Append(',');
                }

                builder.Append("{\"role\":\"");
                builder.Append(EscapeJson(_transcript[i].Role));
                builder.Append("\",\"content\":\"");
                builder.Append(EscapeJson(_transcript[i].Text));
                builder.Append("\"}");
            }

            builder.Append(']');
            return builder.ToString();
        }

        private void AppendTurn(string role, string text, string thoughts = null)
        {
            _transcript.Add(new ChatTurn
            {
                Role = role,
                Text = text ?? string.Empty,
                Thoughts = thoughts ?? string.Empty,
            });
            _transcriptScroll = new Vector2(0f, float.MaxValue);
        }

        private bool CurrentModelSupportsThinking()
        {
            return SupportsThinking(ModelOptions[Mathf.Clamp(selectedModelIndex, 0, ModelOptions.Length - 1)]);
        }

        private static bool SupportsThinking(string modelPath)
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                return false;
            }

            return modelPath.IndexOf("Qwen3", StringComparison.OrdinalIgnoreCase) >= 0 &&
                   modelPath.IndexOf("ASR", StringComparison.OrdinalIgnoreCase) < 0;
        }

        private static string ExtractThinkBlocks(string response, out string thoughts)
        {
            thoughts = string.Empty;
            if (string.IsNullOrEmpty(response))
            {
                return string.Empty;
            }

            const string openTag = "<think>";
            const string closeTag = "</think>";
            var answerBuilder = new StringBuilder(response.Length);
            var thoughtsBuilder = new StringBuilder();
            var index = 0;
            while (index < response.Length)
            {
                var openIndex = response.IndexOf(openTag, index, StringComparison.OrdinalIgnoreCase);
                if (openIndex < 0)
                {
                    answerBuilder.Append(response, index, response.Length - index);
                    break;
                }

                answerBuilder.Append(response, index, openIndex - index);
                var contentStart = openIndex + openTag.Length;
                var closeIndex = response.IndexOf(closeTag, contentStart, StringComparison.OrdinalIgnoreCase);
                if (closeIndex < 0)
                {
                    // Unterminated <think> (streaming/truncation): treat the rest as thoughts.
                    thoughtsBuilder.Append(response, contentStart, response.Length - contentStart);
                    break;
                }

                thoughtsBuilder.Append(response, contentStart, closeIndex - contentStart);
                index = closeIndex + closeTag.Length;
            }

            thoughts = thoughtsBuilder.ToString().Trim();
            return answerBuilder.ToString().Trim();
        }

        private IEnumerator ResolveModelPathCoroutine(
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

            var sourcePath = Path.Combine(Application.streamingAssetsPath, configuredPath).Replace("\\", "/");
            if (!sourcePath.Contains("://"))
            {
                if (!File.Exists(sourcePath))
                {
                    onError(new FileNotFoundException($"Model file not found in StreamingAssets: {sourcePath}"));
                    yield break;
                }

                onSuccess(sourcePath);
                yield break;
            }

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM");
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

            using var request = UnityWebRequest.Get(sourcePath);
            request.downloadHandler = new DownloadHandlerFile(destinationPath)
            {
                removeFileOnAbort = true,
            };
            yield return request.SendWebRequest();
            if (request.result != UnityWebRequest.Result.Success)
            {
                onError(new IOException($"Failed to copy model from StreamingAssets: {request.error}"));
                yield break;
            }

            onSuccess(destinationPath);
        }

        private static string ResolveWindowsExecutablePath(string configuredPath)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
            {
                throw new ArgumentException("Windows executable path is required.", nameof(configuredPath));
            }

            if (Path.IsPathRooted(configuredPath))
            {
                return configuredPath;
            }

            var projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
            if (string.IsNullOrWhiteSpace(projectRoot))
            {
                throw new InvalidOperationException("Failed to resolve Unity project root.");
            }

            var candidateInProject = Path.Combine(projectRoot, configuredPath);
            if (File.Exists(candidateInProject))
            {
                return candidateInProject;
            }

            var repoRoot = Directory.GetParent(projectRoot)?.FullName;
            if (!string.IsNullOrWhiteSpace(repoRoot))
            {
                var candidateInRepo = Path.Combine(repoRoot, configuredPath);
                if (File.Exists(candidateInRepo))
                {
                    return candidateInRepo;
                }
            }

            return candidateInProject;
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

        private static string EscapeJson(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return string.Empty;
            }

            var builder = new StringBuilder(value.Length + 16);
            foreach (var ch in value)
            {
                switch (ch)
                {
                    case '\\':
                        builder.Append("\\\\");
                        break;
                    case '"':
                        builder.Append("\\\"");
                        break;
                    case '\n':
                        builder.Append("\\n");
                        break;
                    case '\r':
                        builder.Append("\\r");
                        break;
                    case '\t':
                        builder.Append("\\t");
                        break;
                    default:
                        if (ch < 0x20)
                        {
                            builder.Append("\\u").Append(((int)ch).ToString("x4"));
                        }
                        else
                        {
                            builder.Append(ch);
                        }

                        break;
                }
            }

            return builder.ToString();
        }

        private static string StripBenchmarkInfo(string response)
        {
            if (string.IsNullOrEmpty(response))
            {
                return string.Empty;
            }

            var benchmarkIndex = response.IndexOf("BenchmarkInfo:", StringComparison.Ordinal);
            return benchmarkIndex < 0 ? response.Trim() : response.Substring(0, benchmarkIndex).Trim();
        }

        private void EnsureClients()
        {
            _client ??= new LiteRtLmUnityClient();
            _windowsCliClient ??= new LiteRtLmWindowsCliClient();
        }

        private struct ChatTurn
        {
            public string Role;
            public string Text;
            public string Thoughts;
            public bool ShowThoughts;
        }
    }
}
