using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// App-side guard and clause-split fallback for the fine-tuned FunctionGemma router
    /// (decl-mode none: the model sees only <see cref="Trigger"/> as the system turn and the
    /// utterance as the user turn, and answers
    /// <c>&lt;start_function_call&gt;call:Name{k:&lt;escape&gt;v&lt;escape&gt;,n:3}&lt;end_function_call&gt;</c>).
    ///
    /// This is a port of <c>Tools/Research/FunctionGemma/FunctionGemmaBench/fg_common.py</c>
    /// (<c>parse_call</c>, <c>validate</c>, <c>grounded</c>, <c>segments</c>,
    /// <c>combine_segments</c>); the measured accuracy in docs is for that exact behaviour, so
    /// change both together. Pure C# on purpose — it runs outside Unity for the self-test.
    /// </summary>
    public static class LiteRtLmFunctionGemmaRouter
    {
        /// <summary>The system turn the decl-mode-none fine-tune was trained with — byte for byte.</summary>
        public const string Trigger = "You are a model that can do function calling with the following functions";

        public const string RefusalName = "DefaultResponse";

        // Mirrors ToolsPrompt.txt (VoiceCommand-WebServer/assets) and fg_common.py. filterNames is a free
        // string in the schema; the twelve names live only in the ToolsPrompt prose.
        private static readonly HashSet<string> FilterNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "드론 시야", "배터리", "팀 연결선", "장애물", "마커", "전장의 안개", "위협", "적 은엄폐",
            "아군 은엄폐", "주요 지형지물", "기동성", "기동로",
        };

        // Spoken forms per value: Sino, native, counter forms, the contract's STT codes (E/이=2, 이오기=2호기 …)
        // and English digits the recogniser emits. ponytail: lenient substring match — it catches invented
        // slots, not misreads.
        private static readonly Dictionary<long, string[]> NumberWords = new Dictionary<long, string[]>
        {
            [1] = new[] { "일", "하나", "한", "첫", "원" },
            [2] = new[] { "이", "둘", "두", "E", "투" },
            [3] = new[] { "삼", "셋", "세", "쓰리", "사모기" },
            [4] = new[] { "사", "넷", "네", "포" },
            [5] = new[] { "오", "다섯", "파이브" },
            [6] = new[] { "육", "여섯" },
            [7] = new[] { "칠", "일곱" },
            [8] = new[] { "팔", "여덟" },
            [9] = new[] { "구", "아홉" },
            [10] = new[] { "십", "열" },
        };

        private static readonly Dictionary<string, string> CallsignKo = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Alpha"] = "알파", ["Bravo"] = "브라보", ["Charlie"] = "찰리", ["Delta"] = "델타", ["Echo"] = "에코",
            ["Foxtrot"] = "폭스트롯", ["Golf"] = "골프", ["Hotel"] = "호텔", ["India"] = "인디아", ["Juliet"] = "줄리",
            ["Kilo"] = "킬로", ["Lima"] = "리마", ["Mike"] = "마이크", ["November"] = "노벰버", ["Oscar"] = "오스카",
            ["Papa"] = "파파", ["Quebec"] = "퀘벡", ["Romeo"] = "로미오", ["Sierra"] = "시에라", ["Tango"] = "탱고",
            ["Uniform"] = "유니폼", ["Victor"] = "빅터", ["Whiskey"] = "위스키", ["X-ray"] = "엑스레이",
            ["Yankee"] = "양키", ["Zulu"] = "줄루",
        };

        private static readonly Regex CallPattern = new Regex(
            @"<start_function_call>call:([A-Za-z_]\w*)\{(.*?)\}<end_function_call>", RegexOptions.Singleline);

        // ponytail: [0-9] where Python's \d also accepts other Unicode digits — the model only emits ASCII.
        private static readonly Regex IntPattern = new Regex(@"^-?[0-9]+$");
        private static readonly Regex FloatPattern = new Regex(@"^-?[0-9]+\.[0-9]*$");

        // ponytail: punctuation-only split — a transcript without punctuation (SenseVoice woitn) is never segmented.
        private static readonly Regex SegmentSplit = new Regex("[,.!?;…]+");

        private const string Escape = "<escape>";

        /// <summary>A function call. Argument values are string, long, double or bool.</summary>
        public sealed class Call
        {
            public string Function;
            public Dictionary<string, object> Arguments = new Dictionary<string, object>();

            public bool IsRefusal => Function == RefusalName;

            /// <summary>Keys sorted, compact — identical to Python's json.dumps(sort_keys=True, separators=(",", ":")).</summary>
            public string ToJson()
            {
                var builder = new StringBuilder("{\"arguments\":{");
                var first = true;
                foreach (var key in Arguments.Keys.OrderBy(k => k, StringComparer.Ordinal))
                {
                    if (!first) builder.Append(',');
                    first = false;
                    AppendJsonString(builder, key);
                    builder.Append(':');
                    AppendJsonValue(builder, Arguments[key]);
                }

                builder.Append("},\"function\":");
                AppendJsonString(builder, Function ?? string.Empty);
                return builder.Append('}').ToString();
            }
        }

        /// <summary>The tool contract, loaded from the OpenAI-style tools JSON (fg_common.load_tools output).</summary>
        public sealed class ToolSchema
        {
            // function → parameter → JSON-schema spec ("type", "enum", …)
            public readonly Dictionary<string, Dictionary<string, Dictionary<string, object>>> Properties =
                new Dictionary<string, Dictionary<string, Dictionary<string, object>>>(StringComparer.Ordinal);
            public readonly Dictionary<string, HashSet<string>> Required =
                new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);

            public static ToolSchema FromOpenAiJson(string json)
            {
                var schema = new ToolSchema();
                foreach (var entry in (List<object>)MiniJson.Parse(json))
                {
                    var function = (Dictionary<string, object>)((Dictionary<string, object>)entry)["function"];
                    var name = (string)function["name"];
                    var parameters = function.TryGetValue("parameters", out var p) ? (Dictionary<string, object>)p : new Dictionary<string, object>();
                    var properties = parameters.TryGetValue("properties", out var props) ? (Dictionary<string, object>)props : new Dictionary<string, object>();
                    schema.Properties[name] = properties.ToDictionary(kv => kv.Key, kv => (Dictionary<string, object>)kv.Value, StringComparer.Ordinal);
                    schema.Required[name] = new HashSet<string>(
                        parameters.TryGetValue("required", out var req) ? ((List<object>)req).Cast<string>() : Enumerable.Empty<string>(),
                        StringComparer.Ordinal);
                }

                return schema;
            }

            public bool Has(string function) => function != null && Properties.ContainsKey(function);
        }

        public static Call Refusal() => new Call { Function = RefusalName };

        /// <summary>The model's raw text → call, or null when it holds no function-call block.</summary>
        public static Call ParseCall(string raw)
        {
            var match = CallPattern.Match(raw ?? string.Empty);
            return match.Success ? new Call { Function = match.Groups[1].Value, Arguments = ParseArgs(match.Groups[2].Value) } : null;
        }

        /// <summary><c>k:&lt;escape&gt;v&lt;escape&gt;,n:3,b:true</c> → arguments. Escaped values may contain ',' and ':'.</summary>
        public static Dictionary<string, object> ParseArgs(string body)
        {
            var args = new Dictionary<string, object>(StringComparer.Ordinal);
            var i = 0;
            while (i < body.Length)
            {
                var colon = body.IndexOf(':', i);
                if (colon < 0) break;

                var key = body.Substring(i, colon - i).Trim().Trim(',').Trim();
                var j = colon + 1;
                if (body.Length - j >= Escape.Length && string.CompareOrdinal(body, j, Escape, 0, Escape.Length) == 0)
                {
                    var end = body.IndexOf(Escape, j + Escape.Length, StringComparison.Ordinal);
                    if (end < 0) end = body.Length;
                    args[key] = body.Substring(j + Escape.Length, end - j - Escape.Length);
                    i = end + Escape.Length;
                }
                else
                {
                    var end = body.IndexOf(',', j);
                    if (end < 0) end = body.Length;
                    args[key] = Scalar(body.Substring(j, end - j));
                    i = end;
                }

                if (i < body.Length && body[i] == ',') i++;
            }

            return args;
        }

        private static object Scalar(string raw)
        {
            raw = raw.Trim();
            if (raw == "true" || raw == "false") return raw == "true";
            if (IntPattern.IsMatch(raw) && long.TryParse(raw, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var n)) return n;
            if (FloatPattern.IsMatch(raw)) return double.Parse(raw, CultureInfo.InvariantCulture);
            return raw; // bare, unescaped string — kept so validation can reject a wrong type
        }

        /// <summary>
        /// The guard: an unparsable, off-schema or (given the utterance) ungrounded call becomes a refusal
        /// instead of an action.
        /// </summary>
        public static Call Validate(Call call, ToolSchema schema, string utterance = null)
        {
            if (call == null || !schema.Has(call.Function)) return Refusal();

            var properties = schema.Properties[call.Function];
            var args = call.Arguments;
            if (!schema.Required[call.Function].All(args.ContainsKey) || !args.Keys.All(properties.ContainsKey)) return Refusal();

            foreach (var pair in args)
            {
                var spec = properties[pair.Key];
                var type = spec.TryGetValue("type", out var t) ? t as string : null;
                if (type == "number" && !(pair.Value is long)) return Refusal();
                if (type == "string" && !(pair.Value is string)) return Refusal();
                if (spec.TryGetValue("enum", out var e) && !((List<object>)e).Any(option => JsonEquals(option, pair.Value))) return Refusal();
                if (pair.Key == "filterNames")
                {
                    var names = SplitFilters(pair.Value as string);
                    if (names.Count == 0 || !names.IsSubsetOf(FilterNames)) return Refusal();
                }
            }

            return utterance != null && !Grounded(args, utterance) ? Refusal() : call;
        }

        /// <summary>Every slot value must be traceable to the utterance — the model may not invent a number, team or name.</summary>
        public static bool Grounded(Dictionary<string, object> args, string utterance)
        {
            var flat = utterance.Replace(" ", string.Empty).ToLowerInvariant();
            foreach (var pair in args)
            {
                if (pair.Value is long number)
                {
                    var spoken = NumberWords.TryGetValue(number, out var words) && words.Any(w => flat.Contains(w.ToLowerInvariant()));
                    if (!flat.Contains(number.ToString(CultureInfo.InvariantCulture)) && !spoken) return false;
                }
                else if (pair.Key == "team")
                {
                    var team = pair.Value as string ?? string.Empty;
                    var korean = CallsignKo.TryGetValue(team, out var ko) ? ko : "\0";
                    if (!flat.Contains(korean) && !flat.Replace("-", string.Empty).Contains(team.ToLowerInvariant().Replace("-", string.Empty))) return false;
                }
                else if (pair.Key == "soldierName" && !flat.Contains((pair.Value as string ?? string.Empty).Replace(" ", string.Empty)))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>Clauses of an utterance, or an empty list when there is only one.</summary>
        public static List<string> Segments(string utterance)
        {
            var parts = SegmentSplit.Split(utterance ?? string.Empty).Select(p => p.Trim()).Where(p => p.Length > 0).ToList();
            return parts.Count > 1 ? parts : new List<string>();
        }

        /// <summary>
        /// A refused utterance falls back to its clauses: execute only if exactly one distinct executable call
        /// comes out (remarks around a command); two or more stay a refusal, as the contract requires for
        /// multiple actions.
        /// </summary>
        public static Call CombineSegments(Call full, IList<Call> segmentCalls)
        {
            var calls = segmentCalls.Where(c => !c.IsRefusal).ToList();
            return calls.Count > 0 && calls.Select(c => c.ToJson()).Distinct().Count() == 1 ? calls[0] : full;
        }

        /// <summary>
        /// The whole route: <paramref name="infer"/> runs the model on one text from a fresh conversation
        /// (system turn = <see cref="Trigger"/>) and returns its raw output. The utterance is tried first;
        /// a refusal with two or more clauses re-runs each clause, one inference per clause, sequentially.
        /// A refusal comes back with empty arguments — the app fills DefaultResponse.text (see <see cref="WithRefusalText"/>).
        /// </summary>
        public static Call Route(string utterance, Func<string, string> infer, ToolSchema schema)
        {
            var full = Validate(ParseCall(infer(utterance)), schema, utterance);
            if (!full.IsRefusal) return full;

            var parts = Segments(utterance);
            return parts.Count == 0 ? full : CombineSegments(full, parts.Select(p => Validate(ParseCall(infer(p)), schema, p)).ToList());
        }

        /// <summary>The contract's DefaultResponse carries the input text; the decl-none model emits <c>{}</c>.</summary>
        public static Call WithRefusalText(Call call, string utterance)
        {
            if (call.IsRefusal) call.Arguments["text"] = utterance ?? string.Empty;
            return call;
        }

        private static HashSet<string> SplitFilters(string value) =>
            new HashSet<string>((value ?? string.Empty).Split(',').Select(p => p.Trim()).Where(p => p.Length > 0), StringComparer.Ordinal);

        private static bool JsonEquals(object a, object b)
        {
            if (a is string || b is string) return a is string && b is string && string.Equals(a as string, b as string, StringComparison.Ordinal);
            if (a is bool || b is bool) return a is bool && b is bool && (bool)a == (bool)b;
            return a != null && b != null && Convert.ToDouble(a, CultureInfo.InvariantCulture) == Convert.ToDouble(b, CultureInfo.InvariantCulture);
        }

        private static void AppendJsonValue(StringBuilder builder, object value)
        {
            switch (value)
            {
                case string s: AppendJsonString(builder, s); break;
                case bool b: builder.Append(b ? "true" : "false"); break;
                case long n: builder.Append(n.ToString(CultureInfo.InvariantCulture)); break;
                case double d: builder.Append(d.ToString("R", CultureInfo.InvariantCulture)); break;
                default: builder.Append("null"); break;
            }
        }

        private static void AppendJsonString(StringBuilder builder, string value)
        {
            builder.Append('"');
            foreach (var c in value)
            {
                switch (c)
                {
                    case '"': builder.Append("\\\""); break;
                    case '\\': builder.Append("\\\\"); break;
                    case '\n': builder.Append("\\n"); break;
                    case '\r': builder.Append("\\r"); break;
                    case '\t': builder.Append("\\t"); break;
                    case '\b': builder.Append("\\b"); break;
                    case '\f': builder.Append("\\f"); break;
                    default:
                        if (c < 0x20) builder.Append("\\u").Append(((int)c).ToString("x4"));
                        else builder.Append(c);
                        break;
                }
            }

            builder.Append('"');
        }

        /// <summary>Just enough JSON for the tools schema: objects, arrays, strings, numbers, true/false/null.</summary>
        private static class MiniJson
        {
            public static object Parse(string json)
            {
                var i = 0;
                var value = ReadValue(json, ref i);
                SkipWhitespace(json, ref i);
                if (i != json.Length) throw new FormatException($"Trailing data at {i} in tools JSON.");
                return value;
            }

            private static object ReadValue(string s, ref int i)
            {
                SkipWhitespace(s, ref i);
                if (i >= s.Length) throw new FormatException("Unexpected end of tools JSON.");
                switch (s[i])
                {
                    case '{':
                        var obj = new Dictionary<string, object>(StringComparer.Ordinal);
                        i++;
                        SkipWhitespace(s, ref i);
                        if (s[i] == '}') { i++; return obj; }
                        while (true)
                        {
                            SkipWhitespace(s, ref i);
                            var key = ReadString(s, ref i);
                            SkipWhitespace(s, ref i);
                            Expect(s, ref i, ':');
                            obj[key] = ReadValue(s, ref i);
                            SkipWhitespace(s, ref i);
                            if (s[i] == ',') { i++; continue; }
                            Expect(s, ref i, '}');
                            return obj;
                        }
                    case '[':
                        var list = new List<object>();
                        i++;
                        SkipWhitespace(s, ref i);
                        if (s[i] == ']') { i++; return list; }
                        while (true)
                        {
                            list.Add(ReadValue(s, ref i));
                            SkipWhitespace(s, ref i);
                            if (s[i] == ',') { i++; continue; }
                            Expect(s, ref i, ']');
                            return list;
                        }
                    case '"':
                        return ReadString(s, ref i);
                    default:
                        var start = i;
                        while (i < s.Length && ",}] \t\r\n".IndexOf(s[i]) < 0) i++;
                        var token = s.Substring(start, i - start);
                        if (token == "true") return true;
                        if (token == "false") return false;
                        if (token == "null") return null;
                        if (long.TryParse(token, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var n)) return n;
                        return double.Parse(token, NumberStyles.Float, CultureInfo.InvariantCulture);
                }
            }

            private static string ReadString(string s, ref int i)
            {
                Expect(s, ref i, '"');
                var builder = new StringBuilder();
                while (s[i] != '"')
                {
                    if (s[i] != '\\') { builder.Append(s[i++]); continue; }
                    i++;
                    var c = s[i++];
                    switch (c)
                    {
                        case 'n': builder.Append('\n'); break;
                        case 'r': builder.Append('\r'); break;
                        case 't': builder.Append('\t'); break;
                        case 'b': builder.Append('\b'); break;
                        case 'f': builder.Append('\f'); break;
                        case 'u': builder.Append((char)Convert.ToInt32(s.Substring(i, 4), 16)); i += 4; break;
                        default: builder.Append(c); break;
                    }
                }

                i++;
                return builder.ToString();
            }

            private static void Expect(string s, ref int i, char c)
            {
                if (i >= s.Length || s[i] != c) throw new FormatException($"Expected '{c}' at {i} in tools JSON.");
                i++;
            }

            private static void SkipWhitespace(string s, ref int i)
            {
                while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
            }
        }
    }
}
