using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Jev sample ("jevlike offline") on the device: a frozen gemma-3-270m-it trunk plus the jevlike
    /// decision head (<see cref="LiteRtLmJevDecider"/>) answering two Jev-style questions —
    /// <c>area</c> (which of 14 UI areas, or none) and <c>noul</c> (is this an executable command?) —
    /// with a probability per option. Type a command in the on-screen field, or push
    /// <c>LiteRtLmJevSmokeTest.config.json</c> with a <c>promptsFile</c> to run a prompt set headless:
    /// one JSON line per prompt goes to <c>LiteRtLmJevSmokeTest.results.jsonl</c>.
    ///
    /// Model files are not packaged: Tools/Windows/Tests/Run-LiteRtLmAndroidJevSmokeTest.ps1 pushes
    /// the trunk .tflite, tokenizer.model and jev_head.json/.bin into persistentDataPath/LiteRTLM/jev
    /// (the native side needs real file paths, not APK entries).
    /// </summary>
    public sealed class LiteRtLmJevSmokeTestRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM JevSmoke]";
        private const string StatusFileName = "LiteRtLmJevSmokeTest.status.txt";
        private const string ConfigFileName = "LiteRtLmJevSmokeTest.config.json";
        private const string ResultsFileName = "LiteRtLmJevSmokeTest.results.jsonl";

        [Serializable]
        private sealed class Config
        {
            public string trunkFile = "LiteRTLM/jev/gemma3_270m_l12_t48_w8.tflite";
            public string tokenizerFile = "LiteRTLM/jev/tokenizer.model";
            public string headFile = "LiteRTLM/jev/jev_head.json";
            public string promptsFile = "";
            public int warmupRuns = 1;
        }

        [Serializable]
        private sealed class PromptResult
        {
            public int index;
            public string prompt;
            public string top;
            public float topProbability;
            public string verdict;
            public float noulYes;
            public int tokens;
            public float tokenizeMs, trunkMs, headMs, wallMs;
            public float[] probabilities;
            public string error;
        }

        [SerializeField] private string[] demoPrompts =
        {
            "알파팀 제어권 회수해", "3번 드론 카메라 띄워", "위협 기동로 해제", "녹화 시작", "오늘 날씨 어때",
        };

        private Config _config = new Config();
        private LiteRtLmJevDecider _decider;
        private string _input = "브라보팀 복귀시켜";
        private string _display = "Loading…";

        private IEnumerator Start()
        {
            Application.runInBackground = true;
            var configPath = Path.Combine(Application.persistentDataPath, ConfigFileName);
            if (File.Exists(configPath))
            {
                JsonUtility.FromJsonOverwrite(File.ReadAllText(configPath), _config);
            }

            WriteStatus("START", $"trunk={_config.trunkFile}, tokenizer={_config.tokenizerFile}, head={_config.headFile}, " +
                                 $"promptsFile={_config.promptsFile}, platform={Application.platform}");
#if UNITY_ANDROID && !UNITY_EDITOR
            yield return null; // let the first frame draw before the trunk compiles
            if (!Initialize())
            {
                yield break;
            }

            if (!string.IsNullOrWhiteSpace(_config.promptsFile))
            {
                yield return RunPromptSet(Resolve(_config.promptsFile));
            }
            else
            {
                foreach (var prompt in demoPrompts) WriteStatus("DEMO", Describe(prompt, _decider.Decide(prompt)));
            }

            WriteStatus("SUCCESS", "done");
            _display = "Ready — type a command.";
#else
            WriteStatus("SKIP", "The Jev trunk runs through the Android bridge only.");
            _display = "Android only: the editor has no in-process LiteRT path for a plain .tflite.";
            yield break;
#endif
        }

        private static string Resolve(string path) =>
            Path.IsPathRooted(path) ? path : Path.Combine(Application.persistentDataPath, path);

        private bool Initialize()
        {
            try
            {
                var started = Time.realtimeSinceStartup;
                _decider = new LiteRtLmJevDecider(Resolve(_config.trunkFile), Resolve(_config.tokenizerFile), Resolve(_config.headFile));
                var first = _decider.Decide(demoPrompts[0]); // compiles the trunk
                WriteStatus("INITIALIZED", $"headLoadAndFirstCallSeconds={Time.realtimeSinceStartup - started:0.###}, " +
                                           $"compileMs={first.CompileMs:0}, options={string.Join("|", _decider.AreaOptions)}");
                for (var i = 0; i < _config.warmupRuns; i++) _decider.Decide(demoPrompts[0]);
                return true;
            }
            catch (Exception exception)
            {
                WriteStatus("FAILURE", exception.ToString());
                _display = exception.Message;
                return false;
            }
        }

        private IEnumerator RunPromptSet(string path)
        {
            var prompts = File.ReadAllLines(path, Encoding.UTF8).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
            var resultsPath = Path.Combine(Application.persistentDataPath, ResultsFileName);
            File.WriteAllText(resultsPath, string.Empty);
            var failures = 0;
            for (var i = 0; i < prompts.Length; i++)
            {
                yield return null;
                var result = new PromptResult { index = i, prompt = prompts[i] };
                var started = Time.realtimeSinceStartup;
                try
                {
                    var d = _decider.Decide(prompts[i]);
                    result.wallMs = (Time.realtimeSinceStartup - started) * 1000f;
                    result.top = d.Options[d.Top];
                    result.topProbability = d.Probabilities[d.Top];
                    result.verdict = d.Verdict;
                    result.noulYes = d.NoulYes;
                    result.tokens = d.Tokens;
                    result.tokenizeMs = d.TokenizeMs;
                    result.trunkMs = d.TrunkMs;
                    result.headMs = d.HeadMs;
                    result.probabilities = d.Probabilities;
                }
                catch (Exception exception)
                {
                    failures++;
                    result.error = exception.Message;
                }

                File.AppendAllText(resultsPath, JsonUtility.ToJson(result) + "\n", Encoding.UTF8);
                WriteStatus("PROMPT", $"{i + 1}/{prompts.Length}: top={result.top} p={result.topProbability:0.000} " +
                                      $"verdict={result.verdict} trunkMs={result.trunkMs:0.0} headMs={result.headMs:0.0} error={result.error}");
            }

            WriteStatus("PROMPT_SET_DONE", $"prompts={prompts.Length}, failures={failures}, results={resultsPath}");
        }

        private static string Describe(string prompt, LiteRtLmJevDecider.Decision d)
        {
            var top3 = d.Probabilities.Select((p, i) => (p, i)).OrderByDescending(x => x.p).Take(3)
                .Select(x => $"{d.Options[x.i]} {x.p:0.00}");
            return $"{prompt} → {string.Join(", ", top3)} | noul(yes)={d.NoulYes:0.00} | {d.Verdict} | " +
                   $"tokens={d.Tokens} trunk={d.TrunkMs:0.0}ms head={d.HeadMs:0.0}ms";
        }

        private void OnGUI()
        {
            var scale = Mathf.Max(1f, Screen.height / 720f);
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));
            GUILayout.BeginArea(new Rect(16, 16, Screen.width / scale - 32, Screen.height / scale - 32));
            GUILayout.Label("Jev sample — jevlike offline (gemma-3-270m trunk + decision head)");
            GUILayout.BeginHorizontal();
            _input = GUILayout.TextField(_input, GUILayout.MinWidth(360));
            GUI.enabled = _decider != null;
            if (GUILayout.Button("Decide", GUILayout.Width(120)))
            {
                try
                {
                    var d = _decider.Decide(_input);
                    var bars = d.Probabilities.Select((p, i) => (p, i)).OrderByDescending(x => x.p).Take(5)
                        .Select(x => $"{d.Options[x.i],-14} {new string('█', Mathf.RoundToInt(x.p * 30))} {x.p:0.00}");
                    _display = $"{string.Join("\n", bars)}\n\nnoul (executable?) yes={d.NoulYes:0.00}   verdict: {d.Verdict}\n" +
                               $"tokens={d.Tokens}  trunk {d.TrunkMs:0.0} ms  head {d.HeadMs:0.0} ms";
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
