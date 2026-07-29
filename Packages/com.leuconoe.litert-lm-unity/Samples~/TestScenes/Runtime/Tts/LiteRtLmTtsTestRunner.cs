using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Text-to-speech test scene. Speaks a preset or typed line through whatever
    /// backend the platform offers via <see cref="LiteRtLmSystemTts"/>, and reports
    /// the backend, the time taken and where the audio came from.
    ///
    /// Platform reality this scene is built to expose rather than hide:
    ///   • Windows/editor — SAPI, and a Korean voice (Microsoft Heami) ships with
    ///     the OS, so this path works with no model files at all.
    ///   • Android — uses the device TTS engine. The Android target device (kona) is an
    ///     AOSP build with no TTS engine and no way to install one, so the scene
    ///     will report the backend as unavailable there. Giving that device a voice
    ///     needs a bundled-model backend implementing <see cref="ILiteRtLmTts"/>;
    ///     see docs/tts-model-research.md.
    /// </summary>
    public sealed class LiteRtLmTtsTestRunner : MonoBehaviour
    {
        private const string LogPrefix = "[LiteRT-LM TTS]";

        private static readonly string[] LanguageLabels = { "한국어 (ko-KR)", "English (en-US)" };
        private static readonly string[] LanguageTags = { "ko-KR", "en-US" };

        private static readonly string[] KoreanPresets =
        {
            "고도 백 미터로 상승합니다.",
            "배터리 잔량 칠십 퍼센트.",
            "임무 지점에 도착했습니다. 촬영을 시작합니다.",
            "경고. 강풍이 감지되었습니다. 고도를 낮춥니다.",
            "귀환을 시작합니다. 예상 소요 시간 삼 분.",
        };

        private static readonly string[] EnglishPresets =
        {
            "Ascending to one hundred meters.",
            "Battery at seventy percent.",
            "Arrived at the waypoint. Starting capture.",
            "Warning. High wind detected. Reducing altitude.",
            "Returning to base. Three minutes remaining.",
        };

        private static readonly string[] BackendLabels = { "System voice", "Supertonic (LiteRT)" };

        [Header("LiteRT-LM TTS Test")]
        [SerializeField] private int selectedLanguageIndex;
        [SerializeField] private int selectedPresetIndex;
        [SerializeField] private int selectedBackendIndex;

        [Header("Supertonic")]
        [Tooltip("Flow-matching steps. 4 is the desktop default: 1.6x faster than 8 " +
                 "with mel correlation 0.984 against it.")]
        [SerializeField] private int supertonicSteps = 4;

        private ILiteRtLmTts _tts;
        private readonly List<string> _log = new List<string>();
        private string _text = string.Empty;
        private string _status = "Idle";
        private LiteRtLmUi.StatusKind _statusKind = LiteRtLmUi.StatusKind.Idle;
        private bool _isBusy;
        private float _startedAt;
        private Vector2 _panelScroll;
        private Vector2 _logScroll;

        private string[] Presets => selectedLanguageIndex == 0 ? KoreanPresets : EnglishPresets;

        private void Awake()
        {
            _text = Presets[0];
            SelectBackend(selectedBackendIndex);
        }

        private void OnDestroy()
        {
            _tts?.Dispose();
            _tts = null;
        }

        /// <summary>
        /// Swaps the backend. Both implement <see cref="ILiteRtLmTts"/>, so nothing
        /// else in the scene changes — which is the point of the interface: on the
        /// target device the system voice does not exist at all (AOSP, no TTS
        /// engine installable), so Supertonic is the only one that speaks there,
        /// while on the desktop it is the other way round.
        /// </summary>
        private void SelectBackend(int index)
        {
            _tts?.Dispose();
            selectedBackendIndex = Mathf.Clamp(index, 0, BackendLabels.Length - 1);
            _tts = selectedBackendIndex == 1
                ? new LiteRtLmSupertonicTts(steps: Mathf.Max(1, supertonicSteps))
                : LiteRtLmSystemTts.Create();

            LiteRtLmUi.AppendLog(
                _log,
                $"[{LiteRtLmUi.Stamp()}] backend={_tts.BackendName} available={_tts.IsAvailable}");
            Debug.Log($"{LogPrefix} backend={_tts.BackendName} available={_tts.IsAvailable}");
        }

        private void OnGUI()
        {
            LiteRtLmUi.BeginScreen("TTS", out var controlRect, out var outputRect);

            LiteRtLmUi.BeginPanel(controlRect, "Text to speech");
            _panelScroll = GUILayout.BeginScrollView(_panelScroll);

            LiteRtLmUi.Section("Backend");
            var backendIndex = GUILayout.Toolbar(
                Mathf.Clamp(selectedBackendIndex, 0, BackendLabels.Length - 1), BackendLabels);
            if (backendIndex != selectedBackendIndex)
            {
                SelectBackend(backendIndex);
            }

            GUILayout.Label($"Backend: {_tts.BackendName}", LiteRtLmUi.Mono);
            if (selectedBackendIndex == 1)
            {
                var steps = Mathf.RoundToInt(GUILayout.HorizontalSlider(supertonicSteps, 1f, 8f));
                if (steps != supertonicSteps)
                {
                    supertonicSteps = steps;
                    SelectBackend(selectedBackendIndex);
                }

                GUILayout.Label($"Flow steps: {supertonicSteps}", LiteRtLmUi.Mono);
            }

            LiteRtLmUi.Status(_status, _statusKind);
            if (_isBusy)
            {
                GUILayout.Label($"Elapsed: {Time.realtimeSinceStartup - _startedAt:0.0}s");
            }

            if (!_tts.IsAvailable)
            {
                GUILayout.Label(
                    "No system voice on this platform. On the AOSP target (no TTS " +
                    "engine installable) this is expected — the device needs a " +
                    "bundled-model backend, see docs/tts-model-research.md.",
                    LiteRtLmUi.WrapLabel);
            }

            LiteRtLmUi.Section("Language");
            var languageIndex = GUILayout.Toolbar(
                Mathf.Clamp(selectedLanguageIndex, 0, LanguageLabels.Length - 1), LanguageLabels);
            if (languageIndex != selectedLanguageIndex)
            {
                selectedLanguageIndex = languageIndex;
                selectedPresetIndex = 0;
                _text = Presets[0];
                Debug.Log($"{LogPrefix} language -> {LanguageTags[selectedLanguageIndex]}");
            }

            var presetIndex = LiteRtLmUi.OptionRow(
                "Preset lines", Mathf.Clamp(selectedPresetIndex, 0, Presets.Length - 1),
                ShortLabels(Presets), controlRect.width - 24f, !_isBusy);
            if (presetIndex != selectedPresetIndex)
            {
                selectedPresetIndex = presetIndex;
                _text = Presets[presetIndex];
            }

            LiteRtLmUi.Section("Text");
            _text = LiteRtLmUi.TextArea(_text, 70f);

            GUILayout.Space(6);
            GUI.enabled = _tts.IsAvailable && !_isBusy && !string.IsNullOrWhiteSpace(_text);
            if (GUILayout.Button("Speak", GUILayout.Height(40)))
            {
                Debug.Log($"{LogPrefix} Speak pressed");
                StartCoroutine(SpeakRoutine(_text));
            }

            GUI.enabled = _isBusy || _tts.IsSpeaking;
            if (GUILayout.Button("Stop"))
            {
                Debug.Log($"{LogPrefix} Stop pressed");
                _tts.Stop();
                SetStatus("Stopped", LiteRtLmUi.StatusKind.Idle);
            }

            GUI.enabled = true;
            GUILayout.Space(6);
            if (GUILayout.Button("Speak every preset"))
            {
                StartCoroutine(SpeakAllRoutine());
            }

            GUILayout.EndScrollView();
            LiteRtLmUi.EndPanel();

            LiteRtLmUi.BeginPanel(outputRect, "Results");
            GUILayout.Label("Backend, timing and audio path", LiteRtLmUi.SectionHeader);
            _logScroll = LiteRtLmUi.LogView(_log, _logScroll, true, outputRect.height - 70f);
            LiteRtLmUi.EndPanel();
        }

        private IEnumerator SpeakRoutine(string text)
        {
            if (_isBusy)
            {
                yield break;
            }

            _isBusy = true;
            _startedAt = Time.realtimeSinceStartup;
            SetStatus("Speaking…", LiteRtLmUi.StatusKind.Busy);
            LiteRtLmUi.AppendLog(_log, $"[{LiteRtLmUi.Stamp()}] say ({LanguageTags[selectedLanguageIndex]}): {text}");

            LiteRtLmTtsResult result = null;
            yield return _tts.Speak(text, LanguageTags[selectedLanguageIndex], r => result = r);

            _isBusy = false;
            if (result != null && result.Success)
            {
                var wav = string.IsNullOrEmpty(result.WavPath) ? "played directly" : result.WavPath;
                LiteRtLmUi.AppendLog(_log, $"    {result.Backend} · {result.Seconds:0.00}s · {wav}");

                // The neural backend reports what the graphs actually did; surface it
                // so a device run can be compared with the desktop bench numbers.
                if (_tts is LiteRtLmSupertonicTts supertonic &&
                    !string.IsNullOrEmpty(supertonic.LastResultJson))
                {
                    var json = supertonic.LastResultJson;
                    var rtf = LiteRtLmSupertonicTts.ExtractJsonNumber(json, "rtf");
                    var audio = LiteRtLmSupertonicTts.ExtractJsonNumber(json, "audioSeconds");
                    var perStep = LiteRtLmSupertonicTts.ExtractJsonNumber(
                        json, "vectorEstimatorPerStepSeconds");
                    var vocoder = LiteRtLmSupertonicTts.ExtractJsonNumber(json, "vocoderSeconds");
                    var latent = LiteRtLmSupertonicTts.ExtractJsonNumber(json, "latentLen");
                    LiteRtLmUi.AppendLog(
                        _log,
                        $"    RTF {rtf:0.000} · audio {audio:0.00}s · latent {latent:0} · " +
                        $"ve/step {perStep:0.000}s · voc {vocoder:0.000}s");
                }

                SetStatus($"Done in {result.Seconds:0.00}s", LiteRtLmUi.StatusKind.Idle);
            }
            else
            {
                var error = result?.Error ?? "no result";
                LiteRtLmUi.AppendLog(_log, $"    FAILED: {error}");
                SetStatus("Failed: " + error, LiteRtLmUi.StatusKind.Error);
                Debug.LogWarning($"{LogPrefix} {error}");
            }
        }

        private IEnumerator SpeakAllRoutine()
        {
            var presets = Presets;
            for (var i = 0; i < presets.Length; i++)
            {
                selectedPresetIndex = i;
                _text = presets[i];
                yield return SpeakRoutine(presets[i]);
                yield return new WaitForSeconds(0.2f);
            }
        }

        private void SetStatus(string status, LiteRtLmUi.StatusKind kind)
        {
            _status = status;
            _statusKind = kind;
        }

        /// <summary>Preset buttons show a trimmed line so the option row stays inside the column.</summary>
        private static string[] ShortLabels(IReadOnlyList<string> lines)
        {
            var labels = new string[lines.Count];
            for (var i = 0; i < lines.Count; i++)
            {
                var line = lines[i];
                labels[i] = line.Length <= 10 ? line : line.Substring(0, 9) + "…";
            }

            return labels;
        }
    }
}
