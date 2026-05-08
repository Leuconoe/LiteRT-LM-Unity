using System;
using System.Collections;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace LiteRTLM.Unity
{
    public class LiteRtLmSampleController : MonoBehaviour
    {
        private enum WindowsCliState
        {
            Idle,
            Ready,
            Running,
            Error,
        }

        private enum PendingWindowsAction
        {
            None,
            Reset,
            Dispose,
        }

        [Header("LiteRT-LM")]
        [SerializeField] private string modelPath = "gemma-4-E2B-it.litertlm";
        [SerializeField] private string systemInstruction = "You are a helpful assistant.";
        [SerializeField] private string prompt = "Say hello from LiteRT-LM running inside Unity.";
        [SerializeField] private string backend = "GPU";
        [SerializeField] private string windowsCliExecutablePath = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
        [SerializeField] private string windowsBackend = "CPU";
        [SerializeField] private float windowsRequestTimeoutSeconds = 30f;
        [SerializeField] private string cacheDir = "";
        [SerializeField] private int maxNumTokens = 0;
        [SerializeField] private int cpuThreads = 0;
        [SerializeField] private bool enableSpeculativeDecoding = true;
        [SerializeField] private bool resetConversationBeforeEachPrompt = true;

        private LiteRtLmUnityClient _client;
        private LiteRtLmWindowsCliClient _windowsCliClient;
        private string _status = "Idle";
        private string _response = "";
        private string _resolvedModelPath = "";
        private string _resolvedWindowsExecutablePath = "";
        private bool _isBusy;
        private WindowsCliState _windowsCliState = WindowsCliState.Idle;
        private PendingWindowsAction _pendingWindowsAction = PendingWindowsAction.None;
        private CancellationTokenSource _windowsRequestCancellationTokenSource;
        private Vector2 _scroll;
        private bool _hasImeTextFieldFocus;

        private void Awake()
        {
            EnsureClients();
        }

        private void OnDestroy()
        {
            CancelWindowsRequest();
            _windowsRequestCancellationTokenSource?.Dispose();
            _windowsRequestCancellationTokenSource = null;
            _client?.Dispose();
            _client = null;
        }

        private void OnGUI()
        {
            EnsureClients();

            GUILayout.BeginArea(new Rect(20, 20, 760, 800), GUI.skin.box);
            GUILayout.Label("LiteRT-LM Unity Sample");

            var canUseAndroidBridge = _client.IsAvailable;
            var canUseWindowsCli = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;

            if (!canUseAndroidBridge && !canUseWindowsCli)
            {
                GUILayout.Label("This sample supports Android device builds and Windows CLI testing.");
                GUILayout.EndArea();
                return;
            }

            GUILayout.Label($"Status: {_status}");
            GUILayout.Label(canUseAndroidBridge
                ? "Mode: Android bridge"
                : canUseWindowsCli
                    ? "Mode: Windows CLI fallback"
                    : "Mode: Unsupported");
            if (canUseAndroidBridge)
            {
                GUILayout.Label($"Initialized: {_client.IsInitialized}");
            }
            else if (canUseWindowsCli)
            {
                GUILayout.Label($"Windows State: {_windowsCliState}");
            }
            if (!string.IsNullOrEmpty(_resolvedModelPath))
            {
                GUILayout.Label($"Resolved Model: {_resolvedModelPath}");
            }

            _hasImeTextFieldFocus = false;
            GUILayout.Label("Model Path (absolute path or StreamingAssets-relative file name)");
            modelPath = DrawImeTextField(modelPath, "LiteRtLmModelPathField");
            GUILayout.Label("System Instruction");
            systemInstruction = DrawImeTextField(systemInstruction, "LiteRtLmSystemInstructionField");
            GUILayout.Label("Prompt");
            prompt = DrawImeTextField(prompt, "LiteRtLmPromptField");
            if (canUseAndroidBridge)
            {
                resetConversationBeforeEachPrompt = GUILayout.Toggle(
                    resetConversationBeforeEachPrompt,
                    "Reset conversation before each prompt");
                enableSpeculativeDecoding = GUILayout.Toggle(
                    enableSpeculativeDecoding,
                    "Enable Gemma 4 MTP speculative decoding");
            }

            if (canUseWindowsCli)
            {
                GUILayout.Label("Windows CLI Executable Path");
                windowsCliExecutablePath = DrawImeTextField(windowsCliExecutablePath, "LiteRtLmWindowsCliExecutablePathField");
                GUILayout.Label("Windows Backend");
                windowsBackend = DrawImeTextField(windowsBackend, "LiteRtLmWindowsBackendField");
                GUILayout.Label("Windows Request Timeout (Seconds)");
                windowsRequestTimeoutSeconds = Mathf.Max(1f, ParseFloatField(windowsRequestTimeoutSeconds));
            }

            if (!_hasImeTextFieldFocus)
            {
                Input.imeCompositionMode = IMECompositionMode.Auto;
            }

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Initialize", GUILayout.Height(40)))
            {
                if (!_isBusy)
                {
                    if (canUseAndroidBridge)
                    {
                        StartCoroutine(InitializeRoutine());
                    }
                    else if (canUseWindowsCli)
                    {
                        StartCoroutine(PrepareWindowsCliRoutine());
                    }
                }
            }

            if (GUILayout.Button("Send Message", GUILayout.Height(40)))
            {
                if (canUseAndroidBridge)
                {
                    TryRun(() =>
                    {
                        if (resetConversationBeforeEachPrompt)
                        {
                            _client.ResetConversation(systemInstruction);
                            Debug.Log("[LiteRT-LM Sample] Reset conversation before sending prompt.");
                        }

                        _response = _client.SendMessage(prompt);
                        _status = resetConversationBeforeEachPrompt
                            ? "Response received after conversation reset"
                            : "Response received";
                    });
                }
                else if (canUseWindowsCli)
                {
                    if (_windowsCliState == WindowsCliState.Running)
                    {
                        _status = "Windows CLI request is already running.";
                    }
                    else if (_windowsCliState != WindowsCliState.Ready)
                    {
                        _status = "Initialize Windows CLI first.";
                    }
                    else if (!_isBusy)
                    {
                        StartCoroutine(SendWindowsCliMessageRoutine());
                    }
                }
            }

            if (canUseWindowsCli && _windowsCliState == WindowsCliState.Running)
            {
                if (GUILayout.Button("Cancel Request", GUILayout.Height(40)))
                {
                    CancelWindowsRequest();
                }
            }

            if (GUILayout.Button("Reset Conversation", GUILayout.Height(40)))
            {
                if (canUseAndroidBridge)
                {
                    TryRun(() =>
                    {
                        _client.ResetConversation(systemInstruction);
                        _status = "Conversation reset";
                        _response = "";
                    });
                }
                else
                {
                    if (_windowsCliState == WindowsCliState.Running)
                    {
                        _pendingWindowsAction = PendingWindowsAction.Reset;
                        _response = "";
                        _status = "Canceling Windows CLI request...";
                        CancelWindowsRequest();
                        return;
                    }

                    _response = "";
                    ApplyPendingWindowsAction(PendingWindowsAction.Reset);
                }
            }

            if (GUILayout.Button("Dispose", GUILayout.Height(40)))
            {
                TryRun(() =>
                {
                    _client.Dispose();
                    _client = new LiteRtLmUnityClient();

                    if (_windowsCliState == WindowsCliState.Running)
                    {
                        _pendingWindowsAction = PendingWindowsAction.Dispose;
                        _status = "Disposing after Windows CLI request stops...";
                        CancelWindowsRequest();
                        return;
                    }

                    ApplyPendingWindowsAction(PendingWindowsAction.Dispose);
                });
            }
            GUILayout.EndHorizontal();

            GUILayout.Space(12);
            GUILayout.Label("Response");
            _scroll = GUILayout.BeginScrollView(_scroll, GUILayout.Height(260));
            GUILayout.TextArea(_response, GUILayout.ExpandHeight(true));
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private IEnumerator InitializeRoutine()
        {
            _isBusy = true;
            _status = "Preparing model...";

            string resolvedPath = null;
            Exception error = null;
            yield return ResolveModelPathCoroutine(
                modelPath,
                path => resolvedPath = path,
                ex => error = ex);

            if (error != null)
            {
                _status = $"Error: {error.Message}";
                Debug.LogException(error);
                _isBusy = false;
                yield break;
            }

            TryRun(() =>
            {
                _resolvedModelPath = resolvedPath;
                _client.Initialize(
                    _resolvedModelPath,
                    backend: backend,
                    cacheDir: cacheDir,
                    maxNumTokens: maxNumTokens,
                    cpuThreads: cpuThreads,
                    enableSpeculativeDecoding: enableSpeculativeDecoding,
                    systemInstruction: systemInstruction);
                _status = "Initialized";
            });

            _isBusy = false;
        }

        private IEnumerator SendWindowsCliMessageRoutine()
        {
            _isBusy = true;
            _windowsCliState = WindowsCliState.Running;
            _status = "Running Windows CLI...";

            ResetWindowsRequestCancellationTokenSource();
            var task = _windowsCliClient.SendMessageAsync(
                _resolvedWindowsExecutablePath,
                _resolvedModelPath,
                prompt,
                windowsBackend,
                TimeSpan.FromSeconds(Mathf.Max(1f, windowsRequestTimeoutSeconds)),
                _windowsRequestCancellationTokenSource.Token);

            while (!task.IsCompleted)
            {
                yield return null;
            }

            if (task.IsFaulted)
            {
                var ex = task.Exception?.GetBaseException() ?? new InvalidOperationException("Windows CLI task failed.");
                if (_pendingWindowsAction != PendingWindowsAction.None)
                {
                    ApplyPendingWindowsAction(_pendingWindowsAction);
                }
                else
                {
                    _windowsCliState = WindowsCliState.Error;
                    _status = $"Error: {ex.Message}";
                }

                Debug.LogException(ex);
                _isBusy = false;
                yield break;
            }

            if (task.IsCanceled)
            {
                if (_pendingWindowsAction != PendingWindowsAction.None)
                {
                    ApplyPendingWindowsAction(_pendingWindowsAction);
                }
                else
                {
                    _windowsCliState = string.IsNullOrEmpty(_resolvedModelPath) || string.IsNullOrEmpty(_resolvedWindowsExecutablePath)
                        ? WindowsCliState.Idle
                        : WindowsCliState.Ready;
                    _status = "Windows CLI request canceled.";
                }

                _isBusy = false;
                yield break;
            }

            if (_pendingWindowsAction != PendingWindowsAction.None)
            {
                ApplyPendingWindowsAction(_pendingWindowsAction);
            }
            else
            {
                _response = task.Result;
                _windowsCliState = WindowsCliState.Ready;
                _status = "Windows CLI response received";
            }

            _isBusy = false;
        }

        private IEnumerator PrepareWindowsCliRoutine()
        {
            _isBusy = true;
            _status = "Preparing Windows CLI test path...";

            string resolvedPath = null;
            Exception error = null;
            yield return ResolveModelPathCoroutine(
                modelPath,
                path => resolvedPath = path,
                ex => error = ex);

            if (error != null)
            {
                _status = $"Error: {error.Message}";
                Debug.LogException(error);
                _isBusy = false;
                yield break;
            }

            TryRun(() =>
            {
                _resolvedModelPath = resolvedPath;
                _resolvedWindowsExecutablePath = ResolveWindowsExecutablePath(windowsCliExecutablePath);
                if (!File.Exists(_resolvedWindowsExecutablePath))
                {
                    throw new FileNotFoundException($"Windows executable not found: {_resolvedWindowsExecutablePath}", _resolvedWindowsExecutablePath);
                }

                _windowsCliState = WindowsCliState.Ready;
                _status = "Windows CLI ready. Press Send Message to run inference.";
            });

            _isBusy = false;
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

            var destinationDirectory = Path.Combine(Application.persistentDataPath, "LiteRTLM");
            Directory.CreateDirectory(destinationDirectory);
            var destinationPath = Path.Combine(destinationDirectory, configuredPath);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath) ?? destinationDirectory);

            if (File.Exists(destinationPath))
            {
                onSuccess(destinationPath);
                yield break;
            }

            var sourcePath = Path.Combine(Application.streamingAssetsPath, configuredPath).Replace("\\", "/");

            if (sourcePath.Contains("://"))
            {
                using var request = UnityWebRequest.Get(sourcePath);
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    onError(new IOException($"Failed to copy model from StreamingAssets: {request.error}"));
                    yield break;
                }

                File.WriteAllBytes(destinationPath, request.downloadHandler.data);
            }
            else
            {
                if (!File.Exists(sourcePath))
                {
                    var fallbackSourcePath = ResolveEditorFallbackModelPath(configuredPath);
                    if (string.IsNullOrEmpty(fallbackSourcePath))
                    {
                        onError(new FileNotFoundException($"Model file not found in StreamingAssets: {sourcePath}"));
                        yield break;
                    }

                    File.Copy(fallbackSourcePath, destinationPath, true);
                    onSuccess(destinationPath);
                    yield break;
                }

                File.Copy(sourcePath, destinationPath, true);
            }

            onSuccess(destinationPath);
        }

        private string ResolveWindowsExecutablePath(string configuredPath)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
            {
                throw new ArgumentException("Windows executable path is required.", nameof(configuredPath));
            }

            if (string.Equals(configuredPath, "litert-lm", StringComparison.OrdinalIgnoreCase))
            {
                configuredPath = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
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

        private string ResolveEditorFallbackModelPath(string configuredPath)
        {
            if (Application.platform != RuntimePlatform.WindowsEditor &&
                Application.platform != RuntimePlatform.WindowsPlayer)
            {
                return string.Empty;
            }

            var projectRoot = Directory.GetParent(Application.dataPath);
            if (projectRoot == null)
            {
                return string.Empty;
            }

            var repoRoot = Directory.GetParent(projectRoot.FullName);
            if (repoRoot == null)
            {
                return string.Empty;
            }

            var runtimeTestdataPath = Path.Combine(repoRoot.FullName, "runtime", "testdata", configuredPath);
            if (File.Exists(runtimeTestdataPath))
            {
                return runtimeTestdataPath;
            }

            var defaultRuntimeModelPath = Path.Combine(repoRoot.FullName, "runtime", "testdata", "test_lm.litertlm");
            if (File.Exists(defaultRuntimeModelPath))
            {
                return defaultRuntimeModelPath;
            }

            return string.Empty;
        }

        private void TryRun(Action action)
        {
            try
            {
                action();
            }
            catch (Exception ex)
            {
                if (Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer)
                {
                    _windowsCliState = WindowsCliState.Error;
                }

                _status = $"Error: {ex.Message}";
                Debug.LogException(ex);
            }
        }

        private float ParseFloatField(float currentValue)
        {
            var currentText = currentValue.ToString("0.##");
            var updatedText = GUILayout.TextField(currentText);
            return float.TryParse(updatedText, out var parsedValue) ? parsedValue : currentValue;
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

        private void ResetWindowsRequestCancellationTokenSource()
        {
            _windowsRequestCancellationTokenSource?.Dispose();
            _windowsRequestCancellationTokenSource = new CancellationTokenSource();
        }

        private void CancelWindowsRequest()
        {
            if (_windowsRequestCancellationTokenSource is { IsCancellationRequested: false })
            {
                _windowsRequestCancellationTokenSource.Cancel();
            }
        }

        private void ApplyPendingWindowsAction(PendingWindowsAction action)
        {
            _pendingWindowsAction = PendingWindowsAction.None;

            switch (action)
            {
                case PendingWindowsAction.Reset:
                    _response = "";
                    if (!string.IsNullOrEmpty(_resolvedModelPath) && !string.IsNullOrEmpty(_resolvedWindowsExecutablePath))
                    {
                        _windowsCliState = WindowsCliState.Ready;
                        _status = "Windows CLI mode has no persistent conversation state. Ready for a new request.";
                    }
                    else
                    {
                        _windowsCliState = WindowsCliState.Idle;
                        _status = "Windows CLI mode has no persistent conversation state.";
                    }

                    break;
                case PendingWindowsAction.Dispose:
                    _resolvedWindowsExecutablePath = "";
                    _resolvedModelPath = "";
                    _response = "";
                    _windowsCliState = WindowsCliState.Idle;
                    _status = "Disposed";
                    break;
                default:
                    break;
            }
        }

        private void EnsureClients()
        {
            _client ??= new LiteRtLmUnityClient();
            _windowsCliClient ??= new LiteRtLmWindowsCliClient();
        }
    }
}
