namespace LiteRTLM.Unity
{
    /// <summary>
    /// The "this voice is synthesized" notice.
    ///
    /// Supertonic's weights are OpenRAIL-M, and use restriction (e) requires that
    /// machine-generated content be disclosed as such. That is a shipping
    /// obligation, not a suggestion, so the text lives here rather than being
    /// retyped per scene — one place to change if legal wants different wording.
    ///
    /// The platform voices carry no model licence, but they are synthesized too,
    /// and a UI that discloses only sometimes is worse than one that always does.
    /// <see cref="IsRequiredFor"/> reports where the obligation is contractual;
    /// showing the notice for every backend is the safer default and costs
    /// nothing.
    /// </summary>
    public static class LiteRtLmTtsDisclosure
    {
        public const string Korean = "합성된 음성입니다.";
        public const string English = "Synthesized voice.";

        /// <summary>
        /// The notice in the language being spoken. <paramref name="language"/> is
        /// the same BCP-47-ish tag <see cref="ILiteRtLmTts.Speak"/> takes; anything
        /// that is not Korean falls back to English.
        /// </summary>
        public static string Notice(string language)
        {
            return !string.IsNullOrEmpty(language) &&
                   language.StartsWith("ko", System.StringComparison.OrdinalIgnoreCase)
                ? Korean
                : English;
        }

        /// <summary>
        /// True when the backend's model licence requires the disclosure.
        ///
        /// Only the neural backend does — the platform voices ship with the OS and
        /// carry no such term. Callers that always display the notice do not need
        /// this; it exists so a compliance check can point at something concrete.
        /// </summary>
        public static bool IsRequiredFor(ILiteRtLmTts tts)
        {
            return tts is LiteRtLmSupertonicTts;
        }
    }
}
