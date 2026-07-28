using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Supertonic's text front end, ported from the MIT reference implementation
    /// (supertone-inc/supertonic, <c>py/helper.py</c> <c>UnicodeProcessor</c>).
    ///
    /// It lives in C# rather than in the native bridge on purpose: the first step
    /// is Unicode NFKD, and doing that in C++ would mean linking ICU into the AAR.
    /// <see cref="string.Normalize(NormalizationForm)"/> is in the BCL and behaves
    /// the same as Python's <c>unicodedata.normalize("NFKD", …)</c>, so the models
    /// receive identical ids without the dependency.
    ///
    /// NFKD matters for Korean specifically: it decomposes each Hangul syllable
    /// into its jamo, which is what the indexer has entries for. Composed
    /// syllables map to -1 (unknown).
    ///
    /// The indexer file is a flat JSON array of 65,536 entries, code point → id.
    /// </summary>
    public sealed class LiteRtLmSupertonicText
    {
        /// <summary>Languages the model was trained with; anything else is rejected.</summary>
        public static readonly IReadOnlyList<string> SupportedLanguages = new[]
        {
            "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
            "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
            "ru", "sk", "sl", "sv", "tr", "uk", "vi",
            // "na" = language-agnostic; the model handles the text without a tag.
            "na",
        };

        private const int IndexerSize = 65536;

        private readonly int[] _indexer;

        private LiteRtLmSupertonicText(int[] indexer) => _indexer = indexer;

        /// <summary>
        /// Parses <c>unicode_indexer.json</c> — a flat array of 65,536 integers.
        /// Hand-parsed rather than run through JsonUtility, which cannot deserialize
        /// a bare top-level array.
        /// </summary>
        public static LiteRtLmSupertonicText FromIndexerJson(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new ArgumentException("indexer json is empty", nameof(json));
            }

            var indexer = new int[IndexerSize];
            var count = 0;
            var value = 0;
            var negative = false;
            var inNumber = false;

            foreach (var c in json)
            {
                if (c == '-' && !inNumber)
                {
                    negative = true;
                    inNumber = true;
                    value = 0;
                }
                else if (c >= '0' && c <= '9')
                {
                    inNumber = true;
                    value = (value * 10) + (c - '0');
                }
                else if (inNumber)
                {
                    if (count < IndexerSize)
                    {
                        indexer[count] = negative ? -value : value;
                    }

                    count++;
                    value = 0;
                    negative = false;
                    inNumber = false;
                }
            }

            if (inNumber && count < IndexerSize)
            {
                indexer[count++] = negative ? -value : value;
            }

            if (count != IndexerSize)
            {
                throw new FormatException(
                    $"unicode_indexer.json holds {count} entries, expected {IndexerSize}.");
            }

            return new LiteRtLmSupertonicText(indexer);
        }

        /// <summary>
        /// Normalizes, tags and maps <paramref name="text"/> to model ids.
        /// The returned mask is all ones and as long as the ids: the reference
        /// implementation only uses it to pad a batch, and we synthesize one
        /// utterance at a time.
        /// </summary>
        public void Encode(string text, string language, out int[] textIds, out float[] textMask)
        {
            var prepared = Preprocess(text, language);
            textIds = new int[prepared.Length];
            textMask = new float[prepared.Length];

            for (var i = 0; i < prepared.Length; i++)
            {
                // Characters above the BMP cannot be indexed; the emoji strip above
                // removes the ones that occur in practice, and anything left maps to
                // unknown rather than throwing mid-utterance.
                var code = prepared[i];
                textIds[i] = code < IndexerSize ? _indexer[code] : -1;
                textMask[i] = 1f;
            }
        }

        /// <summary>Mirrors <c>UnicodeProcessor._preprocess_text</c>, step for step.</summary>
        public static string Preprocess(string text, string language)
        {
            if (text == null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            var tag = string.IsNullOrWhiteSpace(language) ? "na" : language.Trim().ToLowerInvariant();
            if (tag.Length > 2 && tag[2] == '-')
            {
                tag = tag.Substring(0, 2); // "ko-KR" -> "ko"
            }

            if (!Contains(SupportedLanguages, tag))
            {
                throw new ArgumentException($"Unsupported Supertonic language: {language}", nameof(language));
            }

            var normalized = text.Normalize(NormalizationForm.FormKD);
            var builder = new StringBuilder(normalized.Length + 16);

            for (var i = 0; i < normalized.Length; i++)
            {
                var c = normalized[i];

                if (IsEmoji(normalized, i, out var consumed))
                {
                    i += consumed - 1;
                    continue;
                }

                switch (c)
                {
                    // Dashes and quotes the model was not trained on.
                    case '–': // en dash
                    case '‑': // non-breaking hyphen
                    case '—': // em dash
                        builder.Append('-');
                        continue;
                    case '_':
                    case '[':
                    case ']':
                    case '|':
                    case '/':
                    case '#':
                    case '→': // right arrow
                    case '←': // left arrow
                        builder.Append(' ');
                        continue;
                    case '“':
                    case '”':
                        builder.Append('"');
                        continue;
                    case '‘':
                    case '’':
                    case '´':
                    case '`':
                        builder.Append('\'');
                        continue;
                    // Dropped outright.
                    case '♥': // ♥
                    case '☆': // ☆
                    case '♡': // ♡
                    case '©': // ©
                    case '\\':
                        continue;
                    case '@':
                        builder.Append(" at ");
                        continue;
                }

                builder.Append(c);
            }

            var result = builder.ToString();
            result = result.Replace("e.g.,", "for example, ").Replace("i.e.,", "that is, ");

            result = result
                .Replace(" ,", ",").Replace(" .", ".").Replace(" !", "!")
                .Replace(" ?", "?").Replace(" ;", ";").Replace(" :", ":")
                .Replace(" '", "'");

            while (result.Contains("\"\"")) result = result.Replace("\"\"", "\"");
            while (result.Contains("''")) result = result.Replace("''", "'");
            while (result.Contains("``")) result = result.Replace("``", "`");

            result = CollapseWhitespace(result);

            if (result.Length == 0 || !EndsSentence(result[result.Length - 1]))
            {
                result += ".";
            }

            return $"<{tag}>{result}</{tag}>";
        }

        private static bool EndsSentence(char c)
        {
            switch (c)
            {
                case '.': case '!': case '?': case ';': case ':': case ',':
                case '\'': case '"': case ')': case ']': case '}':
                case '…': // …
                case '。': // 。
                case '」': // 」
                case '』': // 』
                case '】': // 】
                case '〉': // 〉
                case '》': // 》
                case '›': // ›
                case '»': // »
                    return true;
                default:
                    return false;
            }
        }

        private static string CollapseWhitespace(string value)
        {
            var builder = new StringBuilder(value.Length);
            var pendingSpace = false;

            foreach (var c in value)
            {
                if (char.IsWhiteSpace(c))
                {
                    pendingSpace = builder.Length > 0;
                    continue;
                }

                if (pendingSpace)
                {
                    builder.Append(' ');
                    pendingSpace = false;
                }

                builder.Append(c);
            }

            return builder.ToString();
        }

        /// <summary>
        /// The reference strips a fixed set of emoji ranges. Surrogate pairs are
        /// decoded first so the astral ranges match, and `consumed` reports how many
        /// UTF-16 units the caller should skip.
        /// </summary>
        private static bool IsEmoji(string text, int index, out int consumed)
        {
            consumed = 1;
            var c = text[index];

            if (char.IsHighSurrogate(c) && index + 1 < text.Length && char.IsLowSurrogate(text[index + 1]))
            {
                consumed = 2;
                var code = char.ConvertToUtf32(c, text[index + 1]);
                return (code >= 0x1F600 && code <= 0x1F64F) ||
                       (code >= 0x1F300 && code <= 0x1F5FF) ||
                       (code >= 0x1F680 && code <= 0x1F6FF) ||
                       (code >= 0x1F700 && code <= 0x1F77F) ||
                       (code >= 0x1F780 && code <= 0x1F7FF) ||
                       (code >= 0x1F800 && code <= 0x1F8FF) ||
                       (code >= 0x1F900 && code <= 0x1F9FF) ||
                       (code >= 0x1FA00 && code <= 0x1FA6F) ||
                       (code >= 0x1FA70 && code <= 0x1FAFF) ||
                       (code >= 0x1F1E6 && code <= 0x1F1FF);
            }

            return (c >= 0x2600 && c <= 0x26FF) || (c >= 0x2700 && c <= 0x27BF);
        }

        private static bool Contains(IReadOnlyList<string> values, string value)
        {
            for (var i = 0; i < values.Count; i++)
            {
                if (string.Equals(values[i], value, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
