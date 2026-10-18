using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// RAG sample on the device: a query embedder (multilingual-e5-small or EmbeddingGemma, fixed-shape
    /// .tflite through <see cref="LiteRtLmTextEncoder"/>), a KorQuAD-dev passage index
    /// (<see cref="LiteRtLmRagIndex"/>) and gemma-4-E2B answering from the top paragraphs.
    /// Type a question in the on-screen field, or push <c>LiteRtLmRagSmokeTest.config.json</c> with a
    /// <c>questionsFile</c> to run headless: one JSON line per question (retrieval hit, embed / retrieve /
    /// LLM timings incl. TTFT, answer) goes to <c>LiteRtLmRagSmokeTest.results.jsonl</c>; answers are scored
    /// against the gold spans on the desktop (Tools/Research/Rag rag_eval.py grade-device).
    ///
    /// Nothing is packaged: Tools/Windows/Tests/Run-LiteRtLmAndroidRagSmokeTest.ps1 pushes the index
    /// folder (rag_eval.py index) and the LLM into persistentDataPath/LiteRTLM/rag.
    /// </summary>
    public sealed class LiteRtLmRagSmokeTestRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM RagSmoke]";
        private const string StatusFileName = "LiteRtLmRagSmokeTest.status.txt";
        private const string ConfigFileName = "LiteRtLmRagSmokeTest.config.json";
        private const string ResultsFileName = "LiteRtLmRagSmokeTest.results.jsonl";

        [Serializable]
        private sealed class Config
        {
            public string embedderFile = "LiteRTLM/rag/e5-small.json";
            public string passagesFile = "LiteRTLM/rag/passages.json";
            public string questionsFile = "";
            public string llmModelFile = "LiteRTLM/rag/gemma-4-E2B-it.litertlm";
            public string llmBackend = "GPU";
            public int maxNumTokens = 2048;
            public int topK = 3;
            public int maxPassageChars = 700;
            public bool skipLlm;
        }

        /// <summary>Written per embedder by rag_eval.py index; paths are relative to its folder.</summary>
        [Serializable]
        private sealed class EmbedderMeta
        {
            public string embedder, queryGraph, tokenizer, index, queryPrefix;
            public int queryLength, dims, idOffset, unkId = -1, bosId = -1, eosId = -1, padId;
            public bool useMask;
            public float desktopHit1, desktopHit3;
        }

        [Serializable]
        private sealed class Question
        {
            public string id, question;
            public int gold;
            public string[] answers;
        }

        [Serializable]
        private sealed class QueryResult
        {
            public int index;
            public string id, question, embedder, llmBackend;
            public int gold, hit1, hit3, tokens;
            public int[] top;
            public float[] scores;
            public float embedMs, tokenizeMs, encoderInvokeMs, retrieveMs, llmMs, ttftSeconds, prefillTps, decodeTps;
            public int prefillTokens, decodeTokens;
            public string answer, error;
        }

        private static readonly Regex TtftPattern = new Regex(@"Time to first token:\s*([0-9.eE+-]+)\s*s");
        private static readonly Regex PrefillPattern = new Regex(@"Last Prefill:\s*(\d+)\s*tokens,\s*([0-9.eE+-]+)");
        private static readonly Regex DecodePattern = new Regex(@"Last Decode:\s*(\d+)\s*tokens,\s*([0-9.eE+-]+)");

        private Config _config = new Config();
        private EmbedderMeta _meta;
        private LiteRtLmTextEncoder _encoder;
        private LiteRtLmRagIndex _index;
        private LiteRtLmUnityClient _llm;
        private string _input = "훈민정음을 창제한 왕은 누구인가?";
        private string _display = "Loading…";

        private IEnumerator Start()
        {
            Application.runInBackground = true;
            var configPath = Path.Combine(Application.persistentDataPath, ConfigFileName);
            if (File.Exists(configPath))
            {
                JsonUtility.FromJsonOverwrite(File.ReadAllText(configPath), _config);
            }

            WriteStatus("START", $"embedder={_config.embedderFile}, llm={_config.llmModelFile} ({_config.llmBackend}), " +
                                 $"topK={_config.topK}, questions={(_config.questionsFile == "" ? "(interactive)" : _config.questionsFile)}");
            yield return null;
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!Initialize())
            {
                yield break;
            }

            if (!string.IsNullOrEmpty(_config.questionsFile))
            {
                yield return RunQuestionSet(Resolve(_config.questionsFile));
                yield break;
            }

            WriteStatus("SUCCESS", "interactive");
            _display = "Ready — ask a question about the KorQuAD paragraphs.";
#else
            WriteStatus("SKIP", "The text encoder and the LLM run through the Android bridge only.");
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
                var metaPath = Resolve(_config.embedderFile);
                var folder = Path.GetDirectoryName(metaPath) ?? Application.persistentDataPath;
                _meta = JsonUtility.FromJson<EmbedderMeta>(File.ReadAllText(metaPath, Encoding.UTF8));
                _encoder = new LiteRtLmTextEncoder(Path.Combine(folder, _meta.queryGraph), Path.Combine(folder, _meta.tokenizer),
                    new LiteRtLmTextEncoder.Spec
                    {
                        queryPrefix = _meta.queryPrefix, idOffset = _meta.idOffset, unkId = _meta.unkId, bosId = _meta.bosId,
                        eosId = _meta.eosId, padId = _meta.padId, useMask = _meta.useMask,
                    });
                _index = LiteRtLmRagIndex.Load(Path.Combine(folder, _meta.index), Resolve(_config.passagesFile));
                var first = _encoder.Encode("워밍업 질문"); // compiles the graph
                var started = Time.realtimeSinceStartup;
                if (!_config.skipLlm)
                {
                    _llm = new LiteRtLmUnityClient();
                    _llm.Initialize(Resolve(_config.llmModelFile), _config.llmBackend, Application.persistentDataPath,
                        _config.maxNumTokens, systemInstruction: LiteRtLmRagIndex.SystemInstruction);
                    _llm.ResetConversation(LiteRtLmRagIndex.SystemInstruction);
                    _llm.SendMessage("문서 1: 서울은 대한민국의 수도이다.\n\n질문: 대한민국의 수도는?\n답:"); // first-turn warm-up
                }

                WriteStatus("INITIALIZED", $"embedder={_meta.embedder}, dims={_index.Dims}, passages={_index.Count}, " +
                                           $"encoderCompileMs={first.CompileMs:0}, llmInitAndWarmupSeconds={Time.realtimeSinceStartup - started:0.##}, " +
                                           $"desktopHit1={_meta.desktopHit1:0.00}, desktopHit3={_meta.desktopHit3:0.00}");
                return true;
            }
            catch (Exception exception)
            {
                WriteStatus("FAILURE", exception.ToString());
                _display = exception.Message;
                return false;
            }
        }

        private QueryResult Ask(string question, int gold)
        {
            var result = new QueryResult { question = question, gold = gold, embedder = _meta.embedder, llmBackend = _config.llmBackend };
            var started = Time.realtimeSinceStartup;
            var e = _encoder.Encode(question);
            result.embedMs = (Time.realtimeSinceStartup - started) * 1000f;
            result.tokens = e.Tokens;
            result.tokenizeMs = e.TokenizeMs;
            result.encoderInvokeMs = e.InvokeMs;

            started = Time.realtimeSinceStartup;
            var top = _index.TopK(e.Vector, Mathf.Max(3, _config.topK));
            result.retrieveMs = (Time.realtimeSinceStartup - started) * 1000f;
            result.top = top.Select(t => t.Index).ToArray();
            result.scores = top.Select(t => t.Score).ToArray();
            result.hit1 = result.top[0] == gold ? 1 : 0;
            result.hit3 = result.top.Take(3).Contains(gold) ? 1 : 0;

            if (_llm == null)
            {
                return result;
            }

            var prompt = LiteRtLmRagIndex.BuildPrompt(question, top.Take(_config.topK).Select(t => _index.Passage(t.Index)), _config.maxPassageChars);
            _llm.ResetConversation(LiteRtLmRagIndex.SystemInstruction);
            started = Time.realtimeSinceStartup;
            result.answer = _llm.SendMessage(prompt)?.Trim();
            result.llmMs = (Time.realtimeSinceStartup - started) * 1000f;
            var bench = _llm.GetLastBenchmark() ?? "";
            var m = TtftPattern.Match(bench);
            if (m.Success) result.ttftSeconds = float.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
            m = PrefillPattern.Match(bench);
            if (m.Success)
            {
                result.prefillTokens = int.Parse(m.Groups[1].Value);
                result.prefillTps = float.Parse(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
            }

            m = DecodePattern.Match(bench);
            if (m.Success)
            {
                result.decodeTokens = int.Parse(m.Groups[1].Value);
                result.decodeTps = float.Parse(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
            }

            return result;
        }

        private IEnumerator RunQuestionSet(string path)
        {
            var questions = File.ReadAllLines(path, Encoding.UTF8).Where(l => !string.IsNullOrWhiteSpace(l))
                .Select(JsonUtility.FromJson<Question>).ToArray();
            var resultsPath = Path.Combine(Application.persistentDataPath, ResultsFileName);
            File.WriteAllText(resultsPath, string.Empty);
            var failures = 0;
            for (var i = 0; i < questions.Length; i++)
            {
                yield return null;
                QueryResult result;
                try
                {
                    result = Ask(questions[i].question, questions[i].gold);
                }
                catch (Exception exception)
                {
                    failures++;
                    result = new QueryResult { question = questions[i].question, gold = questions[i].gold, error = exception.Message };
                }

                result.index = i;
                result.id = questions[i].id;
                File.AppendAllText(resultsPath, JsonUtility.ToJson(result) + "\n", Encoding.UTF8);
                WriteStatus("QUERY", $"{i + 1}/{questions.Length}: hit1={result.hit1} hit3={result.hit3} embedMs={result.embedMs:0.0} " +
                                     $"llmMs={result.llmMs:0} ttft={result.ttftSeconds:0.00}s error={result.error}");
            }

            WriteStatus("PROMPT_SET_DONE", $"questions={questions.Length}, failures={failures}, results={resultsPath}");
            WriteStatus(failures == 0 ? "SUCCESS" : "FAILURE", "done");
        }

        private void OnGUI()
        {
            var scale = Mathf.Max(1f, Screen.height / 720f);
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));
            GUILayout.BeginArea(new Rect(16, 16, Screen.width / scale - 32, Screen.height / scale - 32));
            GUILayout.Label($"RAG sample — {(_meta == null ? "…" : _meta.embedder)} + gemma-4-E2B ({_config.llmBackend}), KorQuAD-dev paragraphs");
            GUILayout.BeginHorizontal();
            _input = GUILayout.TextField(_input, GUILayout.MinWidth(420));
            GUI.enabled = _index != null;
            if (GUILayout.Button("Ask", GUILayout.Width(120)))
            {
                try
                {
                    var r = Ask(_input, -1);
                    var passages = r.top.Take(_config.topK).Select((p, k) =>
                    {
                        var text = _index.Passage(p);
                        return $"[{k + 1}] ({r.scores[k]:0.000}) {text.Substring(0, Mathf.Min(120, text.Length))}…";
                    });
                    _display = $"{string.Join("\n", passages)}\n\n답: {r.answer}\n\n" +
                               $"embed {r.embedMs:0} ms · retrieve {r.retrieveMs:0.0} ms · LLM {r.llmMs:0} ms (TTFT {r.ttftSeconds:0.00} s)";
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

        private void OnDestroy() => _llm?.Dispose();

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
