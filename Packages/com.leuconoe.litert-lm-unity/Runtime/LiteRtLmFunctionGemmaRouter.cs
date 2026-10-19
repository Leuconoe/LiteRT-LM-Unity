using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// App-side guard, clause-split fallback and confirmation policy for a fine-tuned FunctionGemma router
    /// (decl-mode none: the model sees only <see cref="Trigger"/> as the system turn and the utterance as the
    /// user turn, and answers <c>&lt;start_function_call&gt;call:Name{k:&lt;escape&gt;v&lt;escape&gt;,n:3}&lt;end_function_call&gt;</c>).
    ///
    /// Everything contract-specific comes from the tools JSON (<see cref="ToolSchema.FromOpenAiJson"/>):
    /// JSON-schema types, enums, <c>minimum</c>/<c>maximum</c>, and the router keys the model never sees —
    /// <c>"x-grounded": true</c> on a parameter (its value must be traceable to the utterance),
    /// <c>"x-aliases": {value: [spoken forms]}</c> on an enum parameter, <c>"x-confirm": true</c> on a function.
    ///
    /// C# twin of <c>Tools/Research/FunctionGemma/FunctionGemmaBench/fg_common.py</c> (<c>parse_call</c>,
    /// <c>validate</c>, <c>grounded</c>, <c>segments</c>, <c>combine_segments</c>, <c>needs_confirmation</c>); both
    /// run the same golden table (AutomatedTests/Editor/functiongemma_router_golden.json), so change them
    /// together. Pure C# on purpose — it runs outside Unity for the self-test.
    /// </summary>
    public static class LiteRtLmFunctionGemmaRouter
    {
        /// <summary>The system turn the decl-mode-none fine-tune was trained with — byte for byte.</summary>
        public const string Trigger = "You are a model that can do function calling with the following functions";

        public const string RefusalName = "DefaultResponse";

        private static readonly Regex CallPattern = new Regex(
            @"<start_function_call>call:([A-Za-z_]\w*)\{(.*?)\}<end_function_call>", RegexOptions.Singleline);

        // ponytail: [0-9] where Python's \d also accepts other Unicode digits — the model only emits ASCII.
        private static readonly Regex IntPattern = new Regex(@"^-?[0-9]+$");
        private static readonly Regex FloatPattern = new Regex(@"^-?[0-9]+\.[0-9]*$");

        // ponytail: punctuation-only split — a transcript without punctuation is never segmented.
        private static readonly Regex SegmentSplit = new Regex("[,.!?;…]+");

        private const string Escape = "<escape>";

        /// <summary>A function call. Argument values are string, long, double or bool.</summary>
        public sealed class Call
        {
            public string Function;
            public Dictionary<string, object> Arguments = new Dictionary<string, object>();

            /// <summary>Set by the policy overload of <see cref="Route(string, Func{string, string}, ToolSchema, ConfirmationPolicy, double?)"/>:
            /// ask the user before executing. Not part of <see cref="ToJson"/>.</summary>
            public bool NeedsConfirmation;

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

        /// <summary>The tool contract, loaded from the OpenAI-style tools JSON.</summary>
        public sealed class ToolSchema
        {
            // function → parameter → JSON-schema spec ("type", "enum", "minimum", "x-grounded", "x-aliases", …)
            public readonly Dictionary<string, Dictionary<string, Dictionary<string, object>>> Properties =
                new Dictionary<string, Dictionary<string, Dictionary<string, object>>>(StringComparer.Ordinal);
            public readonly Dictionary<string, HashSet<string>> Required =
                new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
            /// <summary>Functions marked <c>"x-confirm": true</c>.</summary>
            public readonly HashSet<string> ConfirmFunctions = new HashSet<string>(StringComparer.Ordinal);

            public static ToolSchema FromOpenAiJson(string json)
            {
                var schema = new ToolSchema();
                foreach (var entry in (List<object>)ParseJson(json))
                {
                    var function = (Dictionary<string, object>)((Dictionary<string, object>)entry)["function"];
                    var name = (string)function["name"];
                    var parameters = function.TryGetValue("parameters", out var p) ? (Dictionary<string, object>)p : new Dictionary<string, object>();
                    var properties = parameters.TryGetValue("properties", out var props) ? (Dictionary<string, object>)props : new Dictionary<string, object>();
                    schema.Properties[name] = properties.ToDictionary(kv => kv.Key, kv => (Dictionary<string, object>)kv.Value, StringComparer.Ordinal);
                    schema.Required[name] = new HashSet<string>(
                        parameters.TryGetValue("required", out var req) ? ((List<object>)req).Cast<string>() : Enumerable.Empty<string>(),
                        StringComparer.Ordinal);
                    if (function.TryGetValue("x-confirm", out var confirm) && confirm is bool c && c) schema.ConfirmFunctions.Add(name);
                }

                return schema;
            }

            public bool Has(string function) => function != null && Properties.ContainsKey(function);
        }

        /// <summary>
        /// Which routed calls must be confirmed before they run: the contract's <c>x-confirm</c> functions, any extra
        /// <see cref="Functions"/> the app adds, and — when <see cref="MinAsrConfidence"/> is set — any executable call
        /// whose transcript came in below it. A fine-tuned router still makes some wrong calls on hard audio, so
        /// irreversible commands should never run from a single utterance. Mirrors fg_common.needs_confirmation.
        /// </summary>
        public sealed class ConfirmationPolicy
        {
            /// <summary>Extra functions to confirm on top of the contract's x-confirm list.</summary>
            public readonly HashSet<string> Functions = new HashSet<string>(StringComparer.Ordinal);

            /// <summary>Below this ASR confidence any executable call needs confirmation. 0 = off.</summary>
            public double MinAsrConfidence;

            public bool Requires(Call call, ToolSchema schema, double? asrConfidence = null)
            {
                if (call == null || call.IsRefusal) return false;
                if (asrConfidence.HasValue && asrConfidence.Value < MinAsrConfidence) return true;
                return Functions.Contains(call.Function) || (schema != null && schema.ConfirmFunctions.Contains(call.Function));
            }
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
                if (pair.Value is long n
                    && ((spec.TryGetValue("minimum", out var min) && n < Convert.ToDouble(min, CultureInfo.InvariantCulture))
                        || (spec.TryGetValue("maximum", out var max) && n > Convert.ToDouble(max, CultureInfo.InvariantCulture))))
                {
                    return Refusal();
                }
            }

            return utterance != null && !Grounded(args, properties, utterance) ? Refusal() : call;
        }

        /// <summary>
        /// Every <c>x-grounded</c> slot value must be traceable to the utterance — the model may not invent one.
        /// ponytail: lenient substring match (일 matches 1 inside 일곱) — it catches invented slots, not misreads.
        /// </summary>
        public static bool Grounded(Dictionary<string, object> args, Dictionary<string, Dictionary<string, object>> properties, string utterance)
        {
            var flat = utterance.Replace(" ", string.Empty).ToLowerInvariant();
            foreach (var pair in args)
            {
                if (!properties.TryGetValue(pair.Key, out var spec) || !(spec.TryGetValue("x-grounded", out var g) && g is bool grounded && grounded)) continue;

                IEnumerable<string> forms;
                if (pair.Value is long number) forms = NumberForms(number);
                else if (pair.Value is string text) forms = new[] { text }.Concat(Aliases(spec, text));
                else return false;

                if (!forms.Any(f => flat.Contains(f.Replace(" ", string.Empty).ToLowerInvariant()))) return false;
            }

            return true;
        }

        private static IEnumerable<string> Aliases(Dictionary<string, object> spec, string value) =>
            spec.TryGetValue("x-aliases", out var a) && a is Dictionary<string, object> table && table.TryGetValue(value, out var list) && list is List<object> forms
                ? forms.OfType<string>()
                : Enumerable.Empty<string>();

        private const string Sino = "영일이삼사오육칠팔구";
        private static readonly string[] NativeOnes = { "", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉" };
        private static readonly string[] NativeTens = { "", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔" };
        private static readonly Dictionary<string, string> Counter = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["하나"] = "한", ["둘"] = "두", ["셋"] = "세", ["넷"] = "네", ["스물"] = "스무",
        };

        /// <summary>Every way an utterance may carry n: digits, Sino-Korean, native, native counter form (fg_common.number_forms).</summary>
        public static IEnumerable<string> NumberForms(long n)
        {
            var forms = new List<string> { n.ToString(CultureInfo.InvariantCulture) };
            if (n >= 0 && n < 10000) forms.Add(SinoKorean((int)n));
            if (n > 0 && n < 100)
            {
                var tens = NativeTens[n / 10];
                var ones = NativeOnes[n % 10];
                forms.Add(tens + ones);
                var counterOnes = Counter.TryGetValue(ones, out var co) ? co : ones;
                var counterTens = ones.Length == 0 && Counter.TryGetValue(tens, out var ct) ? ct : tens;
                forms.Add(counterTens + counterOnes);
            }

            return forms.Where(f => f.Length > 0).Distinct();
        }

        private static string SinoKorean(int n)
        {
            if (n == 0) return "영";
            var builder = new StringBuilder();
            foreach (var (unit, value) in new[] { ("천", 1000), ("백", 100), ("십", 10) })
            {
                var d = n / value;
                n %= value;
                if (d == 0) continue;
                if (d != 1) builder.Append(Sino[d]);
                builder.Append(unit);
            }

            if (n > 0) builder.Append(Sino[n]);
            return builder.ToString();
        }

        /// <summary>Clauses of an utterance, or an empty list when there is only one.</summary>
        public static List<string> Segments(string utterance)
        {
            var parts = SegmentSplit.Split(utterance ?? string.Empty).Select(p => p.Trim()).Where(p => p.Length > 0).ToList();
            return parts.Count > 1 ? parts : new List<string>();
        }

        /// <summary>
        /// A refused utterance falls back to its clauses: execute only if exactly one distinct executable call
        /// comes out (remarks around a command); two or more stay a refusal — one utterance, one action.
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
        /// A refusal comes back with empty arguments (see <see cref="WithRefusalText"/>).
        /// </summary>
        public static Call Route(string utterance, Func<string, string> infer, ToolSchema schema)
        {
            var full = Validate(ParseCall(infer(utterance)), schema, utterance);
            if (!full.IsRefusal) return full;

            var parts = Segments(utterance);
            return parts.Count == 0 ? full : CombineSegments(full, parts.Select(p => Validate(ParseCall(infer(p)), schema, p)).ToList());
        }

        /// <summary>
        /// <see cref="Route(string, Func{string, string}, ToolSchema)"/> plus the confirmation policy: the call comes
        /// back with <see cref="Call.NeedsConfirmation"/> set when <paramref name="policy"/> says so (pass
        /// <paramref name="asrConfidence"/> to gate on the recogniser's confidence). A refusal never needs confirmation.
        /// </summary>
        public static Call Route(string utterance, Func<string, string> infer, ToolSchema schema, ConfirmationPolicy policy, double? asrConfidence = null)
        {
            var call = Route(utterance, infer, schema);
            call.NeedsConfirmation = policy != null && policy.Requires(call, schema, asrConfidence);
            return call;
        }

        /// <summary>Answer to a confirmation prompt: true = yes, false = no, null = neither (the app cancels).</summary>
        public static bool? ParseConfirmation(string answer)
        {
            var flat = (answer ?? string.Empty).Replace(" ", string.Empty).Trim('.', '!', '?', ',', '~');
            // ponytail: exact-word lists — "아니"/"아니오" checked first so "아니오" never reads as a yes.
            if (new[] { "아니", "아니오", "아니요", "취소", "거절", "안돼", "하지마" }.Any(w => flat.StartsWith(w, StringComparison.Ordinal))) return false;
            if (new[] { "예", "네", "응", "확인", "실행", "승인", "맞아" }.Any(w => flat.StartsWith(w, StringComparison.Ordinal))) return true;
            return null;
        }

        /// <summary>For an app contract whose DefaultResponse carries the input text: the decl-none model emits <c>{}</c>.</summary>
        public static Call WithRefusalText(Call call, string utterance)
        {
            if (call.IsRefusal) call.Arguments["text"] = utterance ?? string.Empty;
            return call;
        }

        /// <summary>Just enough JSON for the tools schema and test tables: objects, arrays, strings, numbers, true/false/null.</summary>
        public static object ParseJson(string json) => MiniJson.Parse(json);

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

        private static class MiniJson
        {
            public static object Parse(string json)
            {
                var i = 0;
                var value = ReadValue(json, ref i);
                SkipWhitespace(json, ref i);
                if (i != json.Length) throw new FormatException($"Trailing data at {i} in JSON.");
                return value;
            }

            private static object ReadValue(string s, ref int i)
            {
                SkipWhitespace(s, ref i);
                if (i >= s.Length) throw new FormatException("Unexpected end of JSON.");
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
                if (i >= s.Length || s[i] != c) throw new FormatException($"Expected '{c}' at {i} in JSON.");
                i++;
            }

            private static void SkipWhitespace(string s, ref int i)
            {
                while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
            }
        }
    }
}
