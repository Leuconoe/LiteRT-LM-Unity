using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using Debug = UnityEngine.Debug;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Windows' built-in SAPI voices, driven through a short PowerShell sidecar that
    /// writes a WAV; Unity then plays the file. This machine class already ships a
    /// Korean voice (<c>Microsoft Heami</c>, ko-KR), so the desktop path needs no
    /// model and no download.
    ///
    /// A sidecar rather than a direct call because Unity's Mono profile cannot
    /// reference <c>System.Speech</c> — the same reason the LLM path shells out to a
    /// CLI. Text goes through a UTF-8 file so Korean survives the console encoding.
    /// </summary>
    public sealed class LiteRtLmWindowsSapiTts : ILiteRtLmTts
    {
        private const float SynthesisTimeoutSeconds = 30f;

        private readonly bool _ownsAudioSource;
        private AudioSource _audioSource;
        private Process _process;
        private string _lastError = string.Empty;

        public LiteRtLmWindowsSapiTts(AudioSource audioSource = null)
        {
            if (audioSource != null)
            {
                _audioSource = audioSource;
                return;
            }

            var host = new GameObject("LiteRtLmTtsAudio") { hideFlags = HideFlags.HideAndDontSave };
            UnityEngine.Object.DontDestroyOnLoad(host);
            _audioSource = host.AddComponent<AudioSource>();
            _audioSource.playOnAwake = false;
            _ownsAudioSource = true;
        }

        public string BackendName => "Windows SAPI";

        public bool IsAvailable
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            get { return true; }
#else
            get { return false; }
#endif
        }

        public bool IsSpeaking => _audioSource != null && _audioSource.isPlaying;

        public string LastError => _lastError;

        /// <summary>Where the last synthesis wrote its WAV. Useful for verification runs.</summary>
        public string LastWavPath { get; private set; } = string.Empty;

        public IEnumerator Speak(string text, string language, Action<LiteRtLmTtsResult> onComplete)
        {
            var result = new LiteRtLmTtsResult { Backend = BackendName };
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            if (string.IsNullOrWhiteSpace(text))
            {
                result.Error = "empty text";
                onComplete?.Invoke(result);
                yield break;
            }

            var startedAt = Time.realtimeSinceStartup;
            var stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss-fff");
            var root = Path.Combine(Path.GetTempPath(), "litertlm-tts");
            Directory.CreateDirectory(root);
            var textPath = Path.Combine(root, $"say-{stamp}.txt");
            var wavPath = Path.Combine(root, $"say-{stamp}.wav");

            var synthesisError = StartSynthesis(text, language, textPath, wavPath);
            if (synthesisError != null)
            {
                result.Error = synthesisError;
                _lastError = synthesisError;
                onComplete?.Invoke(result);
                yield break;
            }

            while (_process != null && !_process.HasExited)
            {
                if (Time.realtimeSinceStartup - startedAt > SynthesisTimeoutSeconds)
                {
                    TryKillProcess();
                    result.Error = $"SAPI synthesis timed out after {SynthesisTimeoutSeconds:F0}s";
                    _lastError = result.Error;
                    onComplete?.Invoke(result);
                    yield break;
                }

                yield return null;
            }

            var exitCode = _process?.ExitCode ?? -1;
            var stderr = _process?.StandardError.ReadToEnd() ?? string.Empty;
            _process?.Dispose();
            _process = null;
            TryDelete(textPath);

            if (exitCode != 0 || !File.Exists(wavPath) || new FileInfo(wavPath).Length == 0)
            {
                result.Error = string.IsNullOrWhiteSpace(stderr)
                    ? $"SAPI sidecar exited with {exitCode}"
                    : stderr.Trim();
                _lastError = result.Error;
                onComplete?.Invoke(result);
                yield break;
            }

            LastWavPath = wavPath;
            result.WavPath = wavPath;

            AudioClip clip = null;
            using (var request = UnityWebRequestMultimedia.GetAudioClip("file://" + wavPath, AudioType.WAV))
            {
                yield return request.SendWebRequest();
#if UNITY_2020_1_OR_NEWER
                var failed = request.result != UnityWebRequest.Result.Success;
#else
                var failed = request.isNetworkError || request.isHttpError;
#endif
                if (failed)
                {
                    result.Error = "could not load synthesized WAV: " + request.error;
                    _lastError = result.Error;
                    onComplete?.Invoke(result);
                    yield break;
                }

                clip = DownloadHandlerAudioClip.GetContent(request);
            }

            if (clip == null)
            {
                result.Error = "synthesized WAV decoded to no audio";
                _lastError = result.Error;
                onComplete?.Invoke(result);
                yield break;
            }

            _audioSource.clip = clip;
            _audioSource.Play();
            while (_audioSource.isPlaying)
            {
                yield return null;
            }

            result.Success = true;
            result.Seconds = Time.realtimeSinceStartup - startedAt;
            _lastError = string.Empty;
            onComplete?.Invoke(result);
#else
            result.Error = "Windows SAPI TTS is only available on Windows.";
            onComplete?.Invoke(result);
            yield break;
#endif
        }

        public void Stop()
        {
            if (_audioSource != null && _audioSource.isPlaying)
            {
                _audioSource.Stop();
            }

            TryKillProcess();
        }

        public void Dispose()
        {
            Stop();

            if (_ownsAudioSource && _audioSource != null)
            {
                UnityEngine.Object.Destroy(_audioSource.gameObject);
            }

            _audioSource = null;
        }

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        /// <summary>Returns null when the sidecar started, otherwise the failure reason.</summary>
        private string StartSynthesis(string text, string language, string textPath, string wavPath)
        {
            var culture = LiteRtLmAndroidSystemTts.NormalizeLanguage(language);

            try
            {
                File.WriteAllText(textPath, text, new UTF8Encoding(false));
            }
            catch (Exception exception)
            {
                return "could not stage the text file: " + exception.Message;
            }

            // Select a voice whose culture matches the request; fall back to the
            // system default rather than failing, and let the caller hear the
            // mismatch instead of getting silence.
            var script =
                "$ErrorActionPreference='Stop';" +
                "Add-Type -AssemblyName System.Speech;" +
                $"$t=[IO.File]::ReadAllText('{Escape(textPath)}',[Text.Encoding]::UTF8);" +
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;" +
                $"$v=$s.GetInstalledVoices()|Where-Object {{$_.VoiceInfo.Culture.Name -like '{Escape(culture)}*'}}|Select-Object -First 1;" +
                "if(-not $v){$v=$s.GetInstalledVoices()|Where-Object {$_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq '" +
                Escape(culture.Split('-')[0]) + "'}|Select-Object -First 1};" +
                "if($v){$s.SelectVoice($v.VoiceInfo.Name)};" +
                $"$s.SetOutputToWaveFile('{Escape(wavPath)}');" +
                "$s.Speak($t);$s.Dispose()";

            var startInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command \"" +
                            script.Replace("\"", "\\\"") + "\"",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
            };

            try
            {
                _process = Process.Start(startInfo);
            }
            catch (Exception exception)
            {
                return "could not start the SAPI sidecar: " + exception.Message;
            }

            return _process == null ? "could not start the SAPI sidecar" : null;
        }

        private static string Escape(string value)
        {
            // Single-quoted PowerShell strings only need doubled single quotes.
            return value.Replace("'", "''");
        }
#endif

        private void TryKillProcess()
        {
            try
            {
                if (_process != null && !_process.HasExited)
                {
                    _process.Kill();
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning("[LiteRT-LM TTS] could not stop the SAPI sidecar: " + exception.Message);
            }
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch (Exception) { }
        }
    }
}
