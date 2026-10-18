using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// A fixed-shape text-encoder <c>.tflite</c> run through <c>UnityLiteRtLmBridge.runTextEncoder</c>
    /// (AAR take19+): SentencePiece in native code, the vocabulary's id rule from <see cref="Spec"/>,
    /// one cached CPU/XNNPACK graph. Used by the RAG sample for query embeddings.
    /// </summary>
    public sealed class LiteRtLmTextEncoder
    {
        private const string BridgeClassName = "com.google.ai.edge.litertlm.unity.UnityLiteRtLmBridge";
        private const int HeaderFloats = 4; // [n, tokenizeMs, invokeMs, compileMs]

        /// <summary>
        /// Id rule and prompt prefix. multilingual-e5 (XLM-R): offset 1, unk 3, bos 0, eos 2, pad 1, mask on,
        /// prefix "query: ". EmbeddingGemma: offset 0, bos 2, eos 1, pad 0, no mask,
        /// prefix "task: search result | query: ". Written by Tools/Research/Rag (rag_eval.py index).
        /// </summary>
        [Serializable]
        public sealed class Spec
        {
            public string queryPrefix = "";
            public int idOffset;
            public int unkId = -1;
            public int bosId = -1;
            public int eosId = -1;
            public int padId;
            public bool useMask;
        }

        public readonly struct Result
        {
            public readonly float[] Vector;
            public readonly int Tokens;
            public readonly float TokenizeMs;
            public readonly float InvokeMs;
            public readonly float CompileMs;

            public Result(float[] vector, int tokens, float tokenizeMs, float invokeMs, float compileMs)
            {
                Vector = vector;
                Tokens = tokens;
                TokenizeMs = tokenizeMs;
                InvokeMs = invokeMs;
                CompileMs = compileMs;
            }
        }

        private readonly string _modelPath;
        private readonly string _tokenizerPath;
        private readonly Spec _spec;

        public LiteRtLmTextEncoder(string modelPath, string tokenizerPath, Spec spec)
        {
            _modelPath = modelPath;
            _tokenizerPath = tokenizerPath;
            _spec = spec ?? throw new ArgumentNullException(nameof(spec));
        }

        /// <summary>Embeds <paramref name="text"/> with the spec's prefix; the vector comes back L2-normalised.</summary>
        public Result Encode(string text)
        {
            using var bridge = new AndroidJavaClass(BridgeClassName);
            var raw = bridge.CallStatic<float[]>("runTextEncoder", _modelPath, _tokenizerPath, _spec.queryPrefix + text,
                _spec.idOffset, _spec.unkId, _spec.bosId, _spec.eosId, _spec.padId, _spec.useMask);
            return FromRaw(raw);
        }

        /// <summary>The native layout <c>[n, tokenizeMs, invokeMs, compileMs, output…]</c> → a normalised vector.</summary>
        public static Result FromRaw(float[] raw)
        {
            if (raw == null || raw.Length <= HeaderFloats)
            {
                throw new InvalidOperationException("Text encoder returned no output.");
            }

            var vector = new float[raw.Length - HeaderFloats];
            Array.Copy(raw, HeaderFloats, vector, 0, vector.Length);
            LiteRtLmRagIndex.Normalize(vector);
            return new Result(vector, (int)raw[0], raw[1], raw[2], raw[3]);
        }
    }

    /// <summary>
    /// Passage index for the RAG sample: texts plus precomputed embeddings (binary: int32 N, int32 D, then N·D
    /// little-endian halves — rag_eval.py index), brute-force cosine top-k, and the answer prompt.
    /// </summary>
    public sealed class LiteRtLmRagIndex
    {
        public const string SystemInstruction =
            "당신은 주어진 문서만 근거로 질문에 답하는 도우미입니다. 답은 문서에 나오는 짧은 구절 하나로만 쓰고, " +
            "설명이나 문장을 덧붙이지 마세요. 문서에 답이 없으면 '모름'이라고만 답하세요.";

        [Serializable]
        private sealed class PassageFile
        {
            public string[] passages;
        }

        private readonly float[] _vectors;
        private readonly string[] _passages;

        public int Count { get; }
        public int Dims { get; }

        private LiteRtLmRagIndex(float[] vectors, string[] passages, int count, int dims)
        {
            _vectors = vectors;
            _passages = passages;
            Count = count;
            Dims = dims;
        }

        public static LiteRtLmRagIndex Load(string embeddingsPath, string passagesJsonPath)
        {
            var passages = JsonUtility.FromJson<PassageFile>(File.ReadAllText(passagesJsonPath, Encoding.UTF8)).passages;
            using var reader = new BinaryReader(File.OpenRead(embeddingsPath));
            var count = reader.ReadInt32();
            var dims = reader.ReadInt32();
            if (passages == null || passages.Length != count)
            {
                throw new InvalidDataException($"Index has {count} vectors but {passages?.Length ?? 0} passages.");
            }

            var vectors = new float[count * dims];
            for (var i = 0; i < vectors.Length; i++)
            {
                vectors[i] = Mathf.HalfToFloat(reader.ReadUInt16());
            }

            return new LiteRtLmRagIndex(vectors, passages, count, dims);
        }

        public string Passage(int index) => _passages[index];

        /// <summary>Indices and cosine scores of the <paramref name="k"/> passages closest to a normalised query.</summary>
        public List<(int Index, float Score)> TopK(float[] query, int k)
        {
            if (query.Length != Dims)
            {
                throw new ArgumentException($"Query has {query.Length} dims, index has {Dims}.", nameof(query));
            }

            var best = new List<(int Index, float Score)>(k + 1);
            for (var i = 0; i < Count; i++)
            {
                var score = 0f;
                var offset = i * Dims;
                for (var d = 0; d < Dims; d++)
                {
                    score += _vectors[offset + d] * query[d];
                }

                if (best.Count < k || score > best[best.Count - 1].Score)
                {
                    var at = best.FindIndex(x => score > x.Score);
                    best.Insert(at < 0 ? best.Count : at, (i, score));
                    if (best.Count > k)
                    {
                        best.RemoveAt(k);
                    }
                }
            }

            return best;
        }

        /// <summary>User turn for the answer LLM: the retrieved paragraphs (each capped at <paramref name="maxChars"/>), then the question.</summary>
        public static string BuildPrompt(string question, IEnumerable<string> passages, int maxChars)
        {
            var builder = new StringBuilder();
            var n = 1;
            foreach (var passage in passages)
            {
                var text = passage.Length > maxChars ? passage.Substring(0, maxChars) + "…" : passage;
                builder.Append("문서 ").Append(n++).Append(": ").Append(text).Append("\n\n");
            }

            return builder.Append("질문: ").Append(question).Append("\n답:").ToString();
        }

        public static void Normalize(float[] vector)
        {
            var sum = 0.0;
            foreach (var v in vector)
            {
                sum += v * v;
            }

            var inv = sum > 0 ? (float)(1.0 / Math.Sqrt(sum)) : 0f;
            for (var i = 0; i < vector.Length; i++)
            {
                vector[i] *= inv;
            }
        }
    }
}
