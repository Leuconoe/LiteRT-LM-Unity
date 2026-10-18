using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// SenseVoice-Small on the device through <see cref="LiteRtLmSenseVoice"/>. Push
    /// <c>LiteRtLmSenseVoiceSmokeTest.config.json</c> with a <c>clipsFile</c> (one <c>audio\treference</c> per line,
    /// audio relative to persistentDataPath) to run headless: one JSON line per clip (text, tokens, CTC confidence,
    /// bucket, feat / infer / decode ms) goes to <c>LiteRtLmSenseVoiceSmokeTest.results.jsonl</c>; the first
    /// <c>dumpCount</c> clips also dump feats.f32 / logits.f32 for desktop parity. Scored on the desktop
    /// (Tools/Research/SenseVoice/SenseVoiceBench/sv_device.py). Nothing is packaged:
    /// Tools/Windows/Tests/Run-LiteRtLmAndroidSenseVoiceSmokeTest.ps1 pushes the model folder and the clips.
    /// </summary>
    public sealed class LiteRtLmSenseVoiceSmokeTestRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM SenseVoiceSmoke]";
        private const string StatusFileName = "LiteRtLmSenseVoiceSmokeTest.status.txt";
        private const string ConfigFileName = "LiteRtLmSenseVoiceSmokeTest.config.json";
        private const string ResultsFileName = "LiteRtLmSenseVoiceSmokeTest.results.jsonl";

        [Serializable]
        private sealed class Config
        {
            public string modelDir = "LiteRTLM/sensevoice";
            public string variant = "w8";
            public string language = "ko";
            public string clipsFile = "";
            public int dumpCount;
        }

        [Serializable]
        private sealed class ClipResult
        {
            public int index;
            public string audio, reference, text, bucket, error;
            public int[] tokens;
            public int frames;
            public float meanMaxProb, minTokenProb, audioSeconds, featMs, inferMs, decodeMs, compileMs, wallMs;
        }

        private Config _config = new Config();
        private LiteRtLmSenseVoice _asr;
        private string _input = "LiteRTLM/sensevoice-clips/0.wav";
        private string _display = "Loading…";

        private IEnumerator Start()
        {
            Application.runInBackground = true;
            var configPath = Path.Combine(Application.persistentDataPath, ConfigFileName);
            if (File.Exists(configPath))
            {
                JsonUtility.FromJsonOverwrite(File.ReadAllText(configPath), _config);
            }

            WriteStatus("START", $"model={_config.modelDir} ({_config.variant}), language={_config.language}, " +
                                 $"clips={(_config.clipsFile == "" ? "(interactive)" : _config.clipsFile)}");
            yield return null;
#if UNITY_ANDROID && !UNITY_EDITOR
            _asr = new LiteRtLmSenseVoice(Resolve(_config.modelDir), _config.variant);
            WriteStatus("INITIALIZED", "native graphs compile on the first clip of each bucket");
            if (!string.IsNullOrEmpty(_config.clipsFile))
            {
                yield return RunClipSet(Resolve(_config.clipsFile));
                yield break;
            }

            WriteStatus("SUCCESS", "interactive");
            _display = "Ready — enter an audio path under persistentDataPath.";
#else
            WriteStatus("SKIP", "SenseVoice runs through the Android bridge only.");
            _display = "Android only: the editor has no in-process LiteRT path for a plain .tflite.";
            yield break;
#endif
        }

        private static string Resolve(string path) =>
            Path.IsPathRooted(path) ? path : Path.Combine(Application.persistentDataPath, path);

        private ClipResult Transcribe(string audio, string dumpDir)
        {
            var started = Time.realtimeSinceStartup;
            var r = _asr.Transcribe(Resolve(audio), _config.language, false, dumpDir);
            return new ClipResult
            {
                audio = audio, text = r.text, tokens = r.tokens, bucket = r.bucket, frames = r.frames, error = r.error,
                meanMaxProb = r.confidence?.meanMaxProb ?? 0f, minTokenProb = r.confidence?.minTokenProb ?? 0f,
                audioSeconds = r.audioSeconds, featMs = r.featMs, inferMs = r.inferMs, decodeMs = r.decodeMs,
                compileMs = r.compileMs, wallMs = (Time.realtimeSinceStartup - started) * 1000f,
            };
        }

        private IEnumerator RunClipSet(string path)
        {
            var clips = File.ReadAllLines(path, Encoding.UTF8).Where(l => !string.IsNullOrWhiteSpace(l))
                .Select(l => l.Split('\t')).ToArray();
            var resultsPath = Path.Combine(Application.persistentDataPath, ResultsFileName);
            File.WriteAllText(resultsPath, string.Empty);
            var failures = 0;
            for (var i = 0; i < clips.Length; i++)
            {
                yield return null;
                ClipResult result;
                try
                {
                    var dumpDir = "";
                    if (i < _config.dumpCount)
                    {
                        dumpDir = Path.Combine(Application.persistentDataPath, "LiteRTLM", "sensevoice-dump", i.ToString());
                        Directory.CreateDirectory(dumpDir);
                    }

                    result = Transcribe(clips[i][0], dumpDir);
                    if (!string.IsNullOrEmpty(result.error)) failures++;
                }
                catch (Exception exception)
                {
                    failures++;
                    result = new ClipResult { audio = clips[i][0], error = exception.Message };
                }

                result.index = i;
                result.reference = clips[i].Length > 1 ? clips[i][1] : "";
                File.AppendAllText(resultsPath, JsonUtility.ToJson(result) + "\n", Encoding.UTF8);
                WriteStatus("CLIP", $"{i + 1}/{clips.Length}: bucket={result.bucket} featMs={result.featMs:0.0} " +
                                    $"inferMs={result.inferMs:0} minTokenProb={result.minTokenProb:0.000} error={result.error}");
            }

            WriteStatus("PROMPT_SET_DONE", $"clips={clips.Length}, failures={failures}, results={resultsPath}");
            WriteStatus(failures == 0 ? "SUCCESS" : "FAILURE", "done");
        }

        private void OnGUI()
        {
            var scale = Mathf.Max(1f, Screen.height / 720f);
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));
            GUILayout.BeginArea(new Rect(16, 16, Screen.width / scale - 32, Screen.height / scale - 32));
            GUILayout.Label($"SenseVoice-Small ({_config.variant}, {_config.language}) — LiteRT CPU/XNNPACK");
            GUILayout.BeginHorizontal();
            _input = GUILayout.TextField(_input, GUILayout.MinWidth(420));
            GUI.enabled = _asr != null;
            if (GUILayout.Button("Transcribe", GUILayout.Width(140)))
            {
                try
                {
                    var r = Transcribe(_input, "");
                    _display = !string.IsNullOrEmpty(r.error) ? r.error
                        : $"{r.text}\n\nconfidence: mean {r.meanMaxProb:0.000} · min token {r.minTokenProb:0.000}\n" +
                          $"bucket {r.bucket} · feat {r.featMs:0} ms · infer {r.inferMs:0} ms · total {r.wallMs:0} ms";
                }
                catch (Exception exception)
                {
                    _display = exception.Message;
                }
            }

            GUI.enabled = true;
            GUILayout.EndHorizontal();
            GUILayout.Label(_display);
            GUILayout.EndArea();
        }

        private static void WriteStatus(string phase, string message)
        {
            Debug.Log($"{LogPrefix} {phase}: {message}");
            try
            {
                File.AppendAllText(Path.Combine(Application.persistentDataPath, StatusFileName),
                    $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}");
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"{LogPrefix} failed to write status file: {exception.Message}");
            }
        }
    }
}
