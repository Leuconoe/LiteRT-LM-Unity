using System;
using System.Diagnostics;
using System.IO;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Offline Jev-style decision model ("jevlike offline"): a frozen gemma-3-270m-it trunk
    /// (layers 0..11, exported .tflite, run natively through
    /// <c>UnityLiteRtLmBridge.runJevTrunk</c>) and a jevlike pointer head evaluated here, so the
    /// options can be re-declared at run time. Picks among closed options with probabilities —
    /// it does not fill arguments, so it is not a function-calling replacement.
    /// Head weights and the tensor contract come from Tools/Research/Jev (jev_head.json + .bin).
    /// Android only: the Windows editor has no in-process LiteRT path for a plain .tflite.
    /// </summary>
    public sealed class LiteRtLmJevDecider
    {
        private const string BridgeClassName = "com.google.ai.edge.litertlm.unity.UnityLiteRtLmBridge";
        private const int HeaderFloats = 4; // [n, tokenizeMs, invokeMs, compileMs]

        [Serializable]
        private sealed class TensorEntry
        {
            public string name;
            public int[] shape;
            public long offset;
        }

        [Serializable]
        private sealed class Questions
        {
            public string[] area;
            public string[] noul;
        }

        [Serializable]
        public sealed class Thresholds
        {
            public float run = 0.5f;
            public float ask = 0.35f;
            public float margin = 0.2f;
        }

        [Serializable]
        private sealed class Manifest
        {
            public int width;
            public int rank;
            public float layernorm_eps = 1e-5f;
            public TensorEntry[] tensors;
            public Questions questions;
            public Thresholds thresholds;
        }

        public sealed class Decision
        {
            public string[] Options;
            public float[] Probabilities;
            public int Top;
            public string Verdict; // run | ask | refuse
            public float NoulYes;
            public int Tokens;
            public float TokenizeMs, TrunkMs, CompileMs, HeadMs;
        }

        private readonly string _trunkPath;
        private readonly string _tokenizerPath;
        private readonly int _width, _rank;
        private readonly float _eps;
        private readonly float[] _ctxNormW, _ctxNormB, _key, _value;
        private readonly float[][] _areaQ, _noulQ;

        public string[] AreaOptions { get; }
        public Thresholds Routing { get; }

        /// <param name="trunkPath">exported trunk .tflite (absolute device path; the native side cannot read from the APK)</param>
        /// <param name="tokenizerModelPath">gemma-3 SentencePiece <c>tokenizer.model</c></param>
        /// <param name="headJsonPath"><c>jev_head.json</c>; <c>jev_head.bin</c> sits next to it</param>
        public LiteRtLmJevDecider(string trunkPath, string tokenizerModelPath, string headJsonPath)
        {
            _trunkPath = trunkPath;
            _tokenizerPath = tokenizerModelPath;
            var manifest = JsonUtility.FromJson<Manifest>(File.ReadAllText(headJsonPath));
            var blob = File.ReadAllBytes(Path.ChangeExtension(headJsonPath, ".bin"));
            _width = manifest.width;
            _rank = manifest.rank;
            _eps = manifest.layernorm_eps;
            AreaOptions = manifest.questions.area;
            Routing = manifest.thresholds ?? new Thresholds();

            float[] Tensor(string name)
            {
                var entry = Array.Find(manifest.tensors, t => t.name == name)
                            ?? throw new InvalidDataException($"jev head: missing tensor {name}");
                var count = 1;
                foreach (var d in entry.shape) count *= d;
                var values = new float[count];
                Buffer.BlockCopy(blob, (int)entry.offset, values, 0, count * sizeof(float));
                return values;
            }

            _ctxNormW = Tensor("context_norm.weight");
            _ctxNormB = Tensor("context_norm.bias");
            _key = Tensor("key.weight");
            _value = Tensor("value.weight");
            var optNormW = Tensor("option_norm.weight");
            var optNormB = Tensor("option_norm.bias");
            var query = Tensor("query.weight");
            _areaQ = ProjectOptions(Tensor("area_options"), optNormW, optNormB, query);
            _noulQ = ProjectOptions(Tensor("noul_options"), optNormW, optNormB, query);
        }

        /// <summary>Scores one utterance against the declared `area` options and the `noul` question.</summary>
        public Decision Decide(string text)
        {
            using var bridge = new AndroidJavaClass(BridgeClassName);
            return Decide(bridge.CallStatic<float[]>("runJevTrunk", _trunkPath, _tokenizerPath, text));
        }

        /// <summary>
        /// Head only, on the native layout <c>[n, tokenizeMs, invokeMs, compileMs, (n-1)·width states]</c> —
        /// lets the editor check the C# head against the Python one without a device.
        /// </summary>
        public Decision Decide(float[] raw)
        {
            var stopwatch = Stopwatch.StartNew();
            var tokens = (int)raw[0];
            var rows = tokens - 1; // <bos> is dropped natively
            var keys = new float[rows][];
            var values = new float[rows][];
            var normed = new float[_width];
            for (var r = 0; r < rows; r++)
            {
                LayerNorm(raw, HeaderFloats + r * _width, _ctxNormW, _ctxNormB, normed);
                keys[r] = Project(_key, normed);
                values[r] = Project(_value, normed);
            }

            var area = Score(_areaQ, keys, values);
            var noul = Score(_noulQ, keys, values);
            stopwatch.Stop();

            var top = ArgMax(area, out var second);
            var decision = new Decision
            {
                Options = AreaOptions,
                Probabilities = area,
                Top = top,
                NoulYes = noul[0],
                Tokens = tokens,
                TokenizeMs = raw[1],
                TrunkMs = raw[2],
                CompileMs = raw[3],
                HeadMs = (float)stopwatch.Elapsed.TotalMilliseconds,
            };
            // voice-agent-unity TurnRouting: run >= run; ask >= ask with a margin lead; `none` never runs.
            var p1 = area[top];
            decision.Verdict = AreaOptions[top] == "none" ? "refuse"
                : p1 >= Routing.run ? "run"
                : p1 >= Routing.ask && p1 - area[second] >= Routing.margin ? "ask"
                : "refuse";
            return decision;
        }

        private float[][] ProjectOptions(float[] options, float[] normW, float[] normB, float[] query)
        {
            var count = options.Length / _width;
            var result = new float[count][];
            var normed = new float[_width];
            for (var o = 0; o < count; o++)
            {
                LayerNorm(options, o * _width, normW, normB, normed);
                result[o] = Project(query, normed);
            }

            return result;
        }

        private float[] Score(float[][] queries, float[][] keys, float[][] values)
        {
            var scale = 1f / Mathf.Sqrt(_rank);
            var logits = new float[queries.Length];
            var weights = new float[keys.Length];
            var attended = new float[_rank];
            for (var o = 0; o < queries.Length; o++)
            {
                var q = queries[o];
                for (var r = 0; r < keys.Length; r++) weights[r] = Dot(q, keys[r]) * scale;
                Softmax(weights);
                Array.Clear(attended, 0, _rank);
                for (var r = 0; r < keys.Length; r++)
                {
                    var w = weights[r];
                    var v = values[r];
                    for (var i = 0; i < _rank; i++) attended[i] += w * v[i];
                }

                logits[o] = Dot(q, attended) * scale;
            }

            Softmax(logits);
            return logits;
        }

        private void LayerNorm(float[] source, int offset, float[] weight, float[] bias, float[] destination)
        {
            double mean = 0, variance = 0;
            for (var i = 0; i < _width; i++) mean += source[offset + i];
            mean /= _width;
            for (var i = 0; i < _width; i++)
            {
                var d = source[offset + i] - mean;
                variance += d * d;
            }

            var inv = 1.0 / Math.Sqrt(variance / _width + _eps);
            for (var i = 0; i < _width; i++)
            {
                destination[i] = (float)((source[offset + i] - mean) * inv) * weight[i] + bias[i];
            }
        }

        // nn.Linear without bias: weight is [rank, width].
        private float[] Project(float[] weight, float[] x)
        {
            var y = new float[_rank];
            for (var r = 0; r < _rank; r++)
            {
                var row = r * _width;
                var sum = 0f;
                for (var i = 0; i < _width; i++) sum += weight[row + i] * x[i];
                y[r] = sum;
            }

            return y;
        }

        private static float Dot(float[] a, float[] b)
        {
            var sum = 0f;
            for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
            return sum;
        }

        private static void Softmax(float[] x)
        {
            var max = float.NegativeInfinity;
            foreach (var v in x) max = Math.Max(max, v);
            var sum = 0f;
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = Mathf.Exp(x[i] - max);
                sum += x[i];
            }

            for (var i = 0; i < x.Length; i++) x[i] /= sum;
        }

        private static int ArgMax(float[] x, out int second)
        {
            var best = 0;
            second = x.Length > 1 ? 1 : 0;
            for (var i = 1; i < x.Length; i++)
            {
                if (x[i] > x[best])
                {
                    second = best;
                    best = i;
                }
                else if (i != best && (second == best || x[i] > x[second]))
                {
                    second = i;
                }
            }

            return best;
        }
    }
}
