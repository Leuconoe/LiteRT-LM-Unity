using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmWindowsCliClient
    {
        private const string CpuBackendName = "cpu";
        private const string GpuBackendName = "gpu";
        private const string FallbackLogPrefix = "[gpu→cpu fallback] ";

        private static int s_gpuUnhealthy;

        /// <summary>
        /// True once a GPU-backed run has failed this session; subsequent GPU requests go straight to CPU.
        /// </summary>
        public static bool IsGpuUnhealthy => Volatile.Read(ref s_gpuUnhealthy) != 0;

        /// <summary>
        /// Clears the session GPU-unhealthy flag so the next GPU request is attempted on the GPU again.
        /// </summary>
        public static void ResetGpuHealth()
        {
            Volatile.Write(ref s_gpuUnhealthy, 0);
        }

        public async Task<string> SendMessageAsync(
            string executablePath,
            string modelPath,
            string prompt,
            string backend,
            TimeSpan timeout,
            CancellationToken cancellationToken,
            string systemMessage = "",
            string toolsJson = "",
            string messagesJson = "",
            bool enableConstrainedDecoding = false,
            bool outputMessageJson = false)
        {
            var isGpuRequest = string.Equals(backend?.Trim(), GpuBackendName, StringComparison.OrdinalIgnoreCase);
            if (isGpuRequest && IsGpuUnhealthy)
            {
                LogFallbackWarning("GPU backend was marked unhealthy earlier this session; running on CPU directly.");
                backend = CpuBackendName;
                isGpuRequest = false;
            }

            try
            {
                var output = await RunProcessAsync(
                        executablePath,
                        modelPath,
                        prompt,
                        backend,
                        timeout,
                        cancellationToken,
                        systemMessage,
                        toolsJson,
                        messagesJson,
                        enableConstrainedDecoding,
                        outputMessageJson)
                    .ConfigureAwait(false);

                if (!isGpuRequest || !string.IsNullOrWhiteSpace(output))
                {
                    return output;
                }

                MarkGpuUnhealthy("GPU run produced empty output; retrying once on CPU.");
            }
            catch (Exception ex) when (isGpuRequest &&
                                       !cancellationToken.IsCancellationRequested &&
                                       IsGpuFallbackEligible(ex))
            {
                MarkGpuUnhealthy($"GPU run failed ({ex.GetType().Name}: {ex.Message}); retrying once on CPU.");
            }

            return await RunProcessAsync(
                    executablePath,
                    modelPath,
                    prompt,
                    CpuBackendName,
                    timeout,
                    cancellationToken,
                    systemMessage,
                    toolsJson,
                    messagesJson,
                    enableConstrainedDecoding,
                    outputMessageJson)
                .ConfigureAwait(false);
        }

        private static bool IsGpuFallbackEligible(Exception exception)
        {
            // Non-zero exit code or empty wrapper output surface as InvalidOperationException;
            // driver hangs surface as TimeoutException via the caller-provided timeout.
            return exception is TimeoutException ||
                   exception is InvalidOperationException ||
                   exception is IOException;
        }

        private static void MarkGpuUnhealthy(string reason)
        {
            Volatile.Write(ref s_gpuUnhealthy, 1);
            LogFallbackWarning($"{reason} GPU is marked unhealthy for the rest of this session.");
        }

        private static void LogFallbackWarning(string message)
        {
            UnityEngine.Debug.LogWarning($"{FallbackLogPrefix}{message}");
        }

        public string SendMessage(
            string executablePath,
            string modelPath,
            string prompt,
            string backend,
            string systemMessage = "",
            string toolsJson = "",
            string messagesJson = "",
            bool enableConstrainedDecoding = false,
            bool outputMessageJson = false)
        {
            return Task.Run(() => SendMessageAsync(
                    executablePath,
                    modelPath,
                    prompt,
                    backend,
                    Timeout.InfiniteTimeSpan,
                    CancellationToken.None,
                    systemMessage,
                    toolsJson,
                    messagesJson,
                    enableConstrainedDecoding,
                    outputMessageJson))
                .GetAwaiter()
                .GetResult();
        }

        private static async Task<string> RunProcessAsync(
            string executablePath,
            string modelPath,
            string prompt,
            string backend,
            TimeSpan timeout,
            CancellationToken cancellationToken,
            string systemMessage,
            string toolsJson,
            string messagesJson,
            bool enableConstrainedDecoding,
            bool outputMessageJson)
        {
            if (string.IsNullOrWhiteSpace(executablePath))
            {
                throw new ArgumentException("Windows CLI executable path is required.", nameof(executablePath));
            }

            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("Model path is required.", nameof(modelPath));
            }

            if (!File.Exists(executablePath))
            {
                throw new FileNotFoundException($"Windows executable not found: {executablePath}", executablePath);
            }

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);
            }

            if (string.IsNullOrWhiteSpace(backend))
            {
                throw new ArgumentException("Windows backend is required.", nameof(backend));
            }

            if (timeout != Timeout.InfiniteTimeSpan && timeout <= TimeSpan.Zero)
            {
                throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be greater than zero or infinite.");
            }

            var tempDirectory = GetWorkspaceTempDirectory();
            var promptFilePath = Path.Combine(tempDirectory, $"litertlm-unity-prompt-{Guid.NewGuid():N}.txt");
            File.WriteAllText(promptFilePath, prompt ?? string.Empty);
            var systemMessageFilePath = WriteOptionalTempFile(tempDirectory, "litertlm-unity-system", systemMessage);
            var toolsJsonFilePath = WriteOptionalTempFile(tempDirectory, "litertlm-unity-tools", toolsJson);
            var messagesJsonFilePath = WriteOptionalTempFile(tempDirectory, "litertlm-unity-messages", messagesJson);

            var executableName = Path.GetFileName(executablePath);
            var isMainExecutable = executableName.StartsWith("litert_lm_main", StringComparison.OrdinalIgnoreCase);

            var wrapperScriptPath = Path.Combine(Path.GetDirectoryName(executablePath) ?? string.Empty, "Run-LiteRtLmSample.ps1");
            var usePowerShellWrapper = isMainExecutable && File.Exists(wrapperScriptPath);

            var commandPath = usePowerShellWrapper ? "pwsh" : executablePath;
            var optionalArguments = BuildOptionalArguments(
                systemMessageFilePath,
                toolsJsonFilePath,
                messagesJsonFilePath,
                enableConstrainedDecoding,
                outputMessageJson,
                usePowerShellWrapper);
            var arguments = usePowerShellWrapper
                ? $"-File \"{wrapperScriptPath}\" -Backend {backend.ToLowerInvariant()} -ModelPath \"{modelPath}\" -PromptFilePath \"{promptFilePath}\"{optionalArguments}"
                : isMainExecutable
                    ? $"--backend={backend.ToLowerInvariant()} --model_path=\"{modelPath}\" --input_prompt_file=\"{promptFilePath}\"{optionalArguments}"
                    : $"run \"{modelPath}\" --input_prompt_file \"{promptFilePath}\" --backend {backend.ToLowerInvariant()}";

            var startInfo = new ProcessStartInfo
            {
                FileName = commandPath,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            using var timeoutCts = timeout == Timeout.InfiniteTimeSpan ? null : new CancellationTokenSource(timeout);
            using var linkedCts = timeoutCts == null
                ? CancellationTokenSource.CreateLinkedTokenSource(cancellationToken)
                : CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);
            var effectiveCancellationToken = linkedCts.Token;

            try
            {
                using var process = new Process
                {
                    StartInfo = startInfo,
                    EnableRaisingEvents = false,
                };

                if (!process.Start())
                {
                    throw new InvalidOperationException($"Failed to start process: {commandPath}");
                }

                var stdoutTask = process.StandardOutput.ReadToEndAsync();
                var stderrTask = process.StandardError.ReadToEndAsync();

                using var cancellationRegistration = effectiveCancellationToken.Register(() =>
                {
                    try
                    {
                        if (!process.HasExited)
                        {
                            process.Kill();
                        }
                    }
                    catch (InvalidOperationException)
                    {
                    }
                });

                try
                {
                    while (!process.HasExited)
                    {
                        await Task.Delay(50, effectiveCancellationToken).ConfigureAwait(false);
                    }

                    var outputCompletionTask = Task.WhenAll(stdoutTask, stderrTask);
                    var completedTask = await Task.WhenAny(
                        outputCompletionTask,
                        Task.Delay(TimeSpan.FromSeconds(5), CancellationToken.None)).ConfigureAwait(false);

                    if (completedTask != outputCompletionTask)
                    {
                        throw new IOException(
                            $"Windows CLI process exited but redirected output did not complete. ExitCode={process.ExitCode}.");
                    }

                    await outputCompletionTask.ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (timeoutCts is { IsCancellationRequested: true } && !cancellationToken.IsCancellationRequested)
                {
                    throw new TimeoutException($"Windows CLI inference timed out after {timeout.TotalSeconds:0.#} seconds.");
                }

                var stdout = stdoutTask.Result.Trim();
                var stderr = stderrTask.Result.Trim();
                if (usePowerShellWrapper)
                {
                    stdout = ExtractWrapperOutput(stdout, prompt ?? string.Empty);
                }

                if (process.ExitCode != 0)
                {
                    throw new InvalidOperationException(
                        $"Windows CLI inference failed (backend={backend}). ExitCode={process.ExitCode}. stderr={stderr}");
                }

                if (usePowerShellWrapper && string.IsNullOrWhiteSpace(stdout))
                {
                    throw new InvalidOperationException(
                        $"Windows CLI inference produced no model response (backend={backend}). stderr={stderr}");
                }

                return stdout;
            }
            finally
            {
                DeleteTempFile(promptFilePath);
                DeleteTempFile(systemMessageFilePath);
                DeleteTempFile(toolsJsonFilePath);
                DeleteTempFile(messagesJsonFilePath);
            }
        }

        private static string WriteOptionalTempFile(string tempDirectory, string prefix, string content)
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return string.Empty;
            }

            var path = Path.Combine(tempDirectory, $"{prefix}-{Guid.NewGuid():N}.txt");
            File.WriteAllText(path, content, Encoding.UTF8);
            return path;
        }

        private static string GetWorkspaceTempDirectory()
        {
#if UNITY_EDITOR
            var projectRoot = Path.GetFullPath(Path.Combine(UnityEngine.Application.dataPath, ".."));
            var repoRoot = Directory.GetParent(projectRoot)?.FullName ?? projectRoot;
            var tempDirectory = Path.Combine(repoRoot, "temp", "windows-cli");
#else
            var tempDirectory = Path.Combine(UnityEngine.Application.temporaryCachePath, "LiteRTLM");
#endif
            Directory.CreateDirectory(tempDirectory);
            return tempDirectory;
        }

        private static string BuildOptionalArguments(
            string systemMessageFilePath,
            string toolsJsonFilePath,
            string messagesJsonFilePath,
            bool enableConstrainedDecoding,
            bool outputMessageJson,
            bool usePowerShellWrapper)
        {
            var builder = new StringBuilder();
            AppendOptionalPathArgument(builder, usePowerShellWrapper ? "SystemMessageFilePath" : "system_message_file", systemMessageFilePath, usePowerShellWrapper);
            AppendOptionalPathArgument(builder, usePowerShellWrapper ? "ToolsJsonFilePath" : "tools_json_file", toolsJsonFilePath, usePowerShellWrapper);
            AppendOptionalPathArgument(builder, usePowerShellWrapper ? "MessagesJsonFilePath" : "messages_json_file", messagesJsonFilePath, usePowerShellWrapper);

            if (enableConstrainedDecoding)
            {
                builder.Append(usePowerShellWrapper ? " -EnableConstrainedDecoding" : " --enable_constrained_decoding=true");
            }

            if (outputMessageJson)
            {
                builder.Append(usePowerShellWrapper ? " -OutputMessageJson" : " --output_message_json=true");
            }

            return builder.ToString();
        }

        private static void AppendOptionalPathArgument(StringBuilder builder, string name, string value, bool usePowerShellWrapper)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return;
            }

            builder.Append(usePowerShellWrapper ? $" -{name} \"{value}\"" : $" --{name}=\"{value}\"");
        }

        private static void DeleteTempFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                return;
            }

            try
            {
                const int maxDeleteAttempts = 3;

                for (var attempt = 0; attempt < maxDeleteAttempts; attempt++)
                {
                    if (!File.Exists(path))
                    {
                        break;
                    }

                    try
                    {
                        File.Delete(path);
                        break;
                    }
                    catch (IOException) when (attempt < maxDeleteAttempts - 1)
                    {
                        Thread.Sleep(50);
                    }
                    catch (UnauthorizedAccessException) when (attempt < maxDeleteAttempts - 1)
                    {
                        Thread.Sleep(50);
                    }
                }
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

        private static string ExtractWrapperOutput(string stdout, string prompt)
        {
            if (string.IsNullOrWhiteSpace(stdout))
            {
                return stdout;
            }

            var lines = stdout.Replace("\r\n", "\n").Split('\n');
            var promptLines = (prompt ?? string.Empty).Replace("\r\n", "\n").Split('\n');
            var promptLineIndex = -1;
            var skipSingleSeparatorAfterPrompt = false;
            var builder = new StringBuilder();

            foreach (var rawLine in lines)
            {
                var line = rawLine.TrimEnd();
                if (promptLineIndex >= 0)
                {
                    if (promptLineIndex < promptLines.Length &&
                        string.Equals(line, promptLines[promptLineIndex].TrimEnd(), StringComparison.Ordinal))
                    {
                        promptLineIndex++;
                        skipSingleSeparatorAfterPrompt = promptLineIndex >= promptLines.Length;
                        continue;
                    }

                    if (skipSingleSeparatorAfterPrompt && string.IsNullOrWhiteSpace(line))
                    {
                        promptLineIndex = -1;
                        skipSingleSeparatorAfterPrompt = false;
                        continue;
                    }

                    promptLineIndex = -1;
                    skipSingleSeparatorAfterPrompt = false;
                }

                if (line.StartsWith("[LiteRT-LM]", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (line.StartsWith("input_prompt:", StringComparison.OrdinalIgnoreCase))
                {
                    var echoedFirstPromptLine = line.Substring("input_prompt:".Length).TrimStart();
                    if (promptLines.Length > 0 &&
                        string.Equals(echoedFirstPromptLine, promptLines[0].TrimEnd(), StringComparison.Ordinal))
                    {
                        promptLineIndex = 1;
                        skipSingleSeparatorAfterPrompt = promptLineIndex >= promptLines.Length;
                    }

                    continue;
                }

                if (IsRuntimeLogLine(line))
                {
                    continue;
                }

                if (builder.Length > 0)
                {
                    builder.AppendLine();
                }

                builder.Append(line);
            }

            return builder.ToString().Trim();
        }

        private static bool IsRuntimeLogLine(string line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                return false;
            }

            return line.StartsWith("INFO:", StringComparison.OrdinalIgnoreCase) ||
                   line.StartsWith("WARNING:", StringComparison.OrdinalIgnoreCase) ||
                   line.StartsWith("ERROR:", StringComparison.OrdinalIgnoreCase) ||
                   line.StartsWith("I0000 ", StringComparison.OrdinalIgnoreCase) ||
                   line.StartsWith("W0000 ", StringComparison.OrdinalIgnoreCase) ||
                   line.StartsWith("E0000 ", StringComparison.OrdinalIgnoreCase);
        }
    }
}
