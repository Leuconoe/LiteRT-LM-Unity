using System.Text.RegularExpressions;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Small text helpers shared by the sample scenes: pulling fields out of the
    /// native JSON results and tidying model output for display. Each scene used
    /// to carry its own copy of these.
    /// </summary>
    public static class LiteRtLmUiText
    {
        /// <summary>Reads a string field out of a flat JSON result.</summary>
        public static string ExtractJsonString(string json, string key)
        {
            var match = Regex.Match(
                json ?? string.Empty,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"(?<value>(?:\\\\.|[^\"\\\\])*)\"");
            return match.Success ? Regex.Unescape(match.Groups["value"].Value) : string.Empty;
        }

        /// <summary>Reads a numeric field out of a flat JSON result, or null.</summary>
        public static double? ExtractJsonNumber(string json, string key)
        {
            var match = Regex.Match(
                json ?? string.Empty,
                "\"" + Regex.Escape(key) + "\"\\s*:\\s*(?<value>-?\\d+(?:\\.\\d+)?)");
            return match.Success && double.TryParse(
                match.Groups["value"].Value,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out var value)
                ? value
                : null;
        }

        /// <summary>
        /// Returns the first balanced {...} block as valid JSON. Models frequently
        /// emit Python-style single quotes ({'tool': 'None'}), which no JSON parser
        /// downstream will accept, so quotes are normalized here.
        /// </summary>
        public static string ExtractFirstJsonObject(string text)
        {
            var raw = ExtractFirstJsonObjectRaw(text);
            return raw == null ? null : NormalizeQuotes(raw);
        }

        /// <summary>
        /// Converts single-quoted keys and values to double quotes, leaving any
        /// apostrophes inside double-quoted strings alone.
        /// </summary>
        public static string NormalizeQuotes(string json)
        {
            const char singleQuote = '\'';
            const char doubleQuote = '"';
            const char backslash = '\\';

            if (string.IsNullOrEmpty(json) || json.IndexOf(singleQuote) < 0)
            {
                return json;
            }

            var builder = new System.Text.StringBuilder(json.Length);
            var inDouble = false;
            var escaped = false;

            foreach (var c in json)
            {
                if (escaped)
                {
                    builder.Append(c);
                    escaped = false;
                    continue;
                }

                if (c == backslash)
                {
                    escaped = true;
                    builder.Append(c);
                }
                else if (c == doubleQuote)
                {
                    inDouble = !inDouble;
                    builder.Append(c);
                }
                else if (c == singleQuote)
                {
                    // An apostrophe inside a proper string stays; a quote outside
                    // one is Python-style syntax and becomes a JSON quote.
                    builder.Append(inDouble ? singleQuote : doubleQuote);
                }
                else
                {
                    builder.Append(c);
                }
            }

            return builder.ToString();
        }

        private static string ExtractFirstJsonObjectRaw(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return null;
            }

            var start = text.IndexOf('{');
            if (start < 0)
            {
                return null;
            }

            var depth = 0;
            var inString = false;
            var escaped = false;

            for (var i = start; i < text.Length; i++)
            {
                var c = text[i];

                if (inString)
                {
                    if (escaped)
                    {
                        escaped = false;
                    }
                    else if (c == '\\')
                    {
                        escaped = true;
                    }
                    else if (c == '"')
                    {
                        inString = false;
                    }

                    continue;
                }

                switch (c)
                {
                    case '"':
                        inString = true;
                        break;
                    case '{':
                        depth++;
                        break;
                    case '}':
                        depth--;
                        if (depth == 0)
                        {
                            return text.Substring(start, i - start + 1);
                        }

                        break;
                }
            }

            return null;
        }

        /// <summary>Collapses whitespace so a value fits on one log line.</summary>
        public static string OneLine(string text, int maxLength = 160)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var collapsed = Regex.Replace(text, @"\s+", " ").Trim();
            return collapsed.Length <= maxLength ? collapsed : collapsed.Substring(0, maxLength) + "…";
        }
    }
}
