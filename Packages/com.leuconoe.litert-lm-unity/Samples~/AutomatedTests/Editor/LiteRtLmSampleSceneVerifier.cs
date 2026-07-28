using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Checks the invariants behind the eleven sample-scene fixes.
    ///
    /// These were all found by running the scenes, and every one of them is a
    /// regression risk: a scene asset can silently lose <c>runOnStart</c>, an
    /// audio catalogue entry can drift from the file on disk, a new scene can
    /// skip the shared two-column layout. Re-running the scenes by hand takes
    /// several minutes and a loaded model; this checks the static side in a
    /// second so only genuinely behavioural things need a manual pass.
    ///
    /// Menu: LiteRT-LM/Verify Sample Scenes.
    /// </summary>
    public static class LiteRtLmSampleSceneVerifier
    {
        private const string ReportRelativePath = "Builds/Logs/SampleSceneVerification.txt";

        private static readonly string[] HandDrivenScenes =
        {
            "LiteRtLmSampleScene",
            "LiteRtLmLlmChatTestScene",
            "LiteRtLmAsrTestScene",
            "LiteRtLmMultimodalTestScene",
            "LiteRtLmAsrFunctionCallingTestScene",
            "LiteRtLmMultimodalFunctionCallingTestScene",
            "LiteRtLmTranslateTestScene",
        };

        private static readonly string[] AutomatedScenes =
        {
            "LiteRtLmAndroidSmokeTestScene",
            "LiteRtLmConversationTestScene",
            "LiteRtLmFunctionCallingBenchmarkScene",
        };

        private sealed class Result
        {
            public string Item;
            public string Check;
            public bool Passed;
            public string Detail;
        }

        [MenuItem("LiteRT-LM/Verify Sample Scenes", priority = 40)]
        public static void Verify()
        {
            var results = new List<Result>();

            CheckItem1(results);
            CheckItem2(results);
            CheckItem3(results);
            CheckItem4(results);
            CheckItem5(results);
            CheckItem6(results);
            CheckItem7(results);
            CheckItem8(results);
            CheckItem9(results);
            CheckItem10(results);
            CheckItem0(results);

            var report = Format(results);
            var path = Path.Combine(
                Directory.GetParent(Application.dataPath)?.FullName ?? string.Empty,
                ReportRelativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, report, Encoding.UTF8);

            var failed = results.Count(r => !r.Passed);
            if (failed == 0)
            {
                Debug.Log($"[LiteRT-LM Verify] {results.Count} checks passed.\n{report}");
            }
            else
            {
                Debug.LogError($"[LiteRT-LM Verify] {failed}/{results.Count} checks FAILED.\n{report}");
            }
        }

        private static string Format(IReadOnlyList<Result> results)
        {
            var builder = new StringBuilder();
            builder.AppendLine("LiteRT-LM sample scene verification");
            builder.AppendLine($"Unity {Application.unityVersion}");
            builder.AppendLine();

            foreach (var group in results.GroupBy(r => r.Item))
            {
                builder.AppendLine($"[{group.Key}]");
                foreach (var r in group)
                {
                    builder.AppendLine($"  {(r.Passed ? "PASS" : "FAIL")}  {r.Check}" +
                                       (string.IsNullOrEmpty(r.Detail) ? string.Empty : $" — {r.Detail}"));
                }

                builder.AppendLine();
            }

            var failed = results.Count(r => !r.Passed);
            builder.AppendLine(failed == 0
                ? $"All {results.Count} checks passed."
                : $"{failed} of {results.Count} checks failed.");
            return builder.ToString();
        }

        private static void Add(List<Result> results, string item, string check, bool passed, string detail = "")
        {
            results.Add(new Result { Item = item, Check = check, Passed = passed, Detail = detail });
        }

        // ---- item 1: URP downgrade and the sample split ---------------------

        private static void CheckItem1(List<Result> results)
        {
            const string item = "1 — URP downgrade, automated scenes split out";

            var manifest = ReadProjectFile("Packages/manifest.json");
            var urp = ExtractJsonValue(manifest, "com.unity.render-pipelines.universal");
            Add(results, item, "URP is on the 2022.3 line (14.x)",
                urp != null && urp.StartsWith("14.", StringComparison.Ordinal), $"found {urp ?? "nothing"}");

            var package = ReadProjectFile("Packages/com.leuconoe.litert-lm-unity/package.json");
            var unity = ExtractJsonValue(package, "unity");
            Add(results, item, "package targets Unity 2022.3", unity == "2022.3", $"found {unity ?? "nothing"}");

            foreach (var scene in AutomatedScenes)
            {
                var path = FindScene(scene);
                Add(results, item, $"{scene} lives in the Automated Tests sample",
                    path != null && path.Contains("Automated Tests", StringComparison.OrdinalIgnoreCase),
                    path ?? "scene not found");
            }

            foreach (var scene in HandDrivenScenes)
            {
                var path = FindScene(scene);
                Add(results, item, $"{scene} lives in the Test Scenes sample",
                    path != null && path.Contains("Test Scenes", StringComparison.OrdinalIgnoreCase)
                                 && !path.Contains("Automated Tests", StringComparison.OrdinalIgnoreCase),
                    path ?? "scene not found");
            }
        }

        // ---- item 2: voice FC prompt and input UI ---------------------------

        private static void CheckItem2(List<Result> results)
        {
            const string item = "2 — Voice FC editable prompt and input UI";
            var type = FindRunnerType("LiteRtLmAsrFunctionCallingTestRunner");
            if (type == null)
            {
                Add(results, item, "runner type exists", false, "LiteRtLmAsrFunctionCallingTestRunner not found");
                return;
            }

            Add(results, item, "system prompt is editable state", HasField(type, "_systemPrompt"));
            Add(results, item, "tool list is editable state", HasField(type, "_toolsJson"));

            var sourceField = type.GetField("_inputSource", BindingFlags.NonPublic | BindingFlags.Instance);
            var names = sourceField == null ? Array.Empty<string>() : Enum.GetNames(sourceField.FieldType);
            Add(results, item, "clip, microphone and typed-text sources",
                names.Length >= 3, string.Join(", ", names));
        }

        // ---- item 3: ASR transcript field and audio catalogue ---------------

        private static void CheckItem3(List<Result> results)
        {
            const string item = "3 — ASR transcript field, no wav-path log";
            var type = FindRunnerType("LiteRtLmAsrTestRunner");
            if (type == null)
            {
                Add(results, item, "runner type exists", false, "LiteRtLmAsrTestRunner not found");
                return;
            }

            Add(results, item, "running transcript is kept apart from the metrics log",
                HasField(type, "_transcriptOnly"));

            // The catalogue drifted from disk once already: an entry named a file
            // that had since been renamed, and selecting it just failed.
            var options = type.GetField("AudioOptions", BindingFlags.NonPublic | BindingFlags.Static)
                ?.GetValue(null) as string[];
            if (options == null)
            {
                Add(results, item, "audio catalogue is readable", false);
                return;
            }

            var missing = options
                .Where(o => !File.Exists(Path.Combine(Application.streamingAssetsPath, o)))
                .ToArray();
            Add(results, item, $"all {options.Length} audio entries exist on disk",
                missing.Length == 0, missing.Length == 0 ? string.Empty : string.Join("; ", missing));
        }

        // ---- item 4: conversation scene ------------------------------------

        private static void CheckItem4(List<Result> results)
        {
            const string item = "4 — Conversation scene camera, UI and model path";
            CheckRunsOnStart(results, item, "LiteRtLmConversationTestScene");
            Add(results, item, "runner attaches the camera and log overlay",
                SourceContains("LiteRtLmConversationTestRunner.cs", "LiteRtLmRunLogOverlay.Attach"));
            Add(results, item, "model path self-heals through StreamingAssets.Resolve",
                SourceContains("LiteRtLmConversationTestRunner.cs", "LiteRtLmStreamingAssets.Resolve"));
        }

        // ---- item 5: benchmark progress ------------------------------------

        private static void CheckItem5(List<Result> results)
        {
            const string item = "5 — Benchmark scene shows progress";
            CheckRunsOnStart(results, item, "LiteRtLmFunctionCallingBenchmarkScene");
            Add(results, item, "per-case status lines reach the overlay",
                SourceContains("LiteRtLmFunctionCallingBenchmarkRunner.cs", "LiteRtLmRunLogOverlay.Current?.LogRaw"));
            Add(results, item, "status file writes are serialised across threads",
                SourceContains("LiteRtLmFunctionCallingBenchmarkRunner.cs", "lock (StatusFileGate)"));
            Add(results, item, "model path self-heals",
                SourceContains("LiteRtLmFunctionCallingBenchmarkRunner.cs", "LiteRtLmStreamingAssets.Resolve"));

            // The log is written from a worker thread while OnGUI reads it.
            Add(results, item, "log buffer hands out a snapshot under a lock",
                SourceContains("LiteRtLmLog.cs", "_snapshot"));
        }

        // ---- item 6: chat composer stays reachable --------------------------

        private static void CheckItem6(List<Result> results)
        {
            const string item = "6 — Chat composer stays reachable";
            Add(results, item, "chat uses the shared two-column screen",
                SourceContains("LiteRtLmLlmChatTestRunner.cs", "LiteRtLmUi.BeginScreen"));
            Add(results, item, "transcript is a separate panel, not stacked above the composer",
                SourceContains("LiteRtLmLlmChatTestRunner.cs", "private void DrawTranscript"));
            Add(results, item, "no fixed-size GUILayout.BeginArea remains",
                !SourceContains("LiteRtLmLlmChatTestRunner.cs", "GUILayout.BeginArea"));
        }

        // ---- item 7: multimodal FC is user driven ---------------------------

        private static void CheckItem7(List<Result> results)
        {
            const string item = "7 — Multimodal FC is user driven";
            const string file = "LiteRtLmMultimodalFunctionCallingTestRunner.cs";

            Add(results, item, "utterance, prompt and tool list are editable",
                SourceContains(file, "_utterance") &&
                SourceContains(file, "_systemPrompt") &&
                SourceContains(file, "_toolsJson"));
            Add(results, item, "image reaches the model as a real [image:] tag",
                SourceContains(file, "[image:") || SourceContains(file, "imagePath"));

            foreach (var tool in new[]
                     { "HandleFruit", "HandlePerson", "HandleCartoon", "HandleAppliance", "HandleAnimal" })
            {
                Add(results, item, $"tool {tool} is offered", SourceContains(file, tool));
            }
        }

        // ---- item 8: multimodal image picker, mic, non-wav audio ------------

        private static void CheckItem8(List<Result> results)
        {
            const string item = "8 — Multimodal file picker, mic and audio decoding";
            const string file = "LiteRtLmMultimodalTestRunner.cs";

            Add(results, item, "image file picker is wired", SourceContains(file, "OpenFileDialog"));
            Add(results, item, "microphone capture is available", SourceContains(file, "LiteRtLmMicVadCapture"));
            Add(results, item, "path fields cannot widen the control column",
                SourceContains(file, "LiteRtLmUi.PathRow"));

            // The media tag decodes wav only; mp3/ogg used to be passed through as
            // plain text, which is why selecting audio looked like it did nothing.
            Add(results, item, "non-wav audio is converted before tagging",
                SourceContains("LiteRtLmDesktopAsr.cs", "EnsureWav"));
            Add(results, item, "media paths are copied to an ASCII, space-free name",
                SourceContains("LiteRtLmDesktopAsr.cs", "SafeMediaPath"));
        }

        // ---- item 9: quick start model path ---------------------------------

        private static void CheckItem9(List<Result> results)
        {
            const string item = "9 — Quick start model path";
            Add(results, item, "model path self-heals",
                SourceContains("LiteRtLmSampleController.cs", "LiteRtLmStreamingAssets.Resolve"));
            Add(results, item, "failure names what is actually available",
                SourceContains("LiteRtLmSampleController.cs", "DescribeAvailable"));
        }

        // ---- item 10: translate ---------------------------------------------

        private static void CheckItem10(List<Result> results)
        {
            const string item = "10 — Translate enabled on desktop, mic transcript shown";
            const string file = "LiteRtLmTranslateTestRunner.cs";

            Add(results, item, "Translate is enabled by the desktop fallback, not only the Android bridge",
                SourceContains(file, "CanUseDesktopFallback"));
            Add(results, item, "the microphone path runs ASR rather than only writing a wav",
                SourceContains(file, "TranslateRoutine"));
            Add(results, item, "results report transcript and translation",
                SourceContains(file, "transcript=") && SourceContains(file, "translation="));
        }

        // ---- item 0: layout unification -------------------------------------

        private static void CheckItem0(List<Result> results)
        {
            const string item = "0 — layout unification";

            foreach (var scene in HandDrivenScenes)
            {
                var path = FindScene(scene);
                if (path == null)
                {
                    Add(results, item, $"{scene} found", false);
                    continue;
                }

                // Solid dark background everywhere: one scene was still on Skybox
                // and read as a different app.
                var text = File.ReadAllText(path);
                var clearFlags = ExtractYamlInt(text, "m_ClearFlags");
                Add(results, item, $"{scene} camera clears to a solid colour",
                    clearFlags == 2, clearFlags.HasValue ? $"m_ClearFlags: {clearFlags}" : "no camera in scene");
            }

            foreach (var file in new[]
                     {
                         "LiteRtLmSampleController.cs", "LiteRtLmLlmChatTestRunner.cs",
                         "LiteRtLmAsrTestRunner.cs", "LiteRtLmMultimodalTestRunner.cs",
                         "LiteRtLmAsrFunctionCallingTestRunner.cs",
                         "LiteRtLmMultimodalFunctionCallingTestRunner.cs",
                         "LiteRtLmTranslateTestRunner.cs",
                     })
            {
                Add(results, item, $"{file} uses the shared screen",
                    SourceContains(file, "LiteRtLmUi.BeginScreen"));
            }

            // Backends are a closed set; a free text field accepted typos that
            // only failed later inside the native call.
            foreach (var file in new[]
                     {
                         "LiteRtLmSampleController.cs", "LiteRtLmLlmChatTestRunner.cs",
                         "LiteRtLmMultimodalTestRunner.cs",
                         "LiteRtLmMultimodalFunctionCallingTestRunner.cs",
                         "LiteRtLmAsrFunctionCallingTestRunner.cs", "LiteRtLmAsrTestRunner.cs",
                     })
            {
                Add(results, item, $"{file} picks the backend from a dropdown",
                    SourceContains(file, "LiteRtLmUi.Dropdown"));
            }
        }

        // ---- helpers ---------------------------------------------------------

        private static void CheckRunsOnStart(List<Result> results, string item, string sceneName)
        {
            var path = FindScene(sceneName);
            if (path == null)
            {
                Add(results, item, $"{sceneName} found", false);
                return;
            }

            var runOnStart = ExtractYamlInt(File.ReadAllText(path), "runOnStart");
            Add(results, item, $"{sceneName} runs on Play",
                runOnStart == 1,
                runOnStart.HasValue ? $"runOnStart: {runOnStart}" : "runOnStart not serialised");
        }

        private static string FindScene(string sceneName)
        {
            return AssetDatabase.FindAssets($"{sceneName} t:Scene")
                .Select(AssetDatabase.GUIDToAssetPath)
                .FirstOrDefault(p => Path.GetFileNameWithoutExtension(p) == sceneName);
        }

        private static Type FindRunnerType(string typeName)
        {
            return AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType($"LiteRTLM.Unity.{typeName}"))
                .FirstOrDefault(t => t != null);
        }

        private static bool HasField(Type type, string name)
        {
            return type.GetField(name, BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public) != null;
        }

        private static bool SourceContains(string fileName, string needle)
        {
            var path = AssetDatabase.FindAssets($"{Path.GetFileNameWithoutExtension(fileName)} t:MonoScript")
                .Select(AssetDatabase.GUIDToAssetPath)
                .FirstOrDefault(p => Path.GetFileName(p) == fileName);
            return path != null && File.ReadAllText(path).Contains(needle, StringComparison.Ordinal);
        }

        private static string ReadProjectFile(string relativePath)
        {
            var root = Directory.GetParent(Application.dataPath)?.FullName;
            if (root == null)
            {
                return string.Empty;
            }

            var path = Path.Combine(root, relativePath);
            return File.Exists(path) ? File.ReadAllText(path) : string.Empty;
        }

        private static string ExtractJsonValue(string json, string key)
        {
            var match = System.Text.RegularExpressions.Regex.Match(
                json ?? string.Empty,
                "\"" + System.Text.RegularExpressions.Regex.Escape(key) + "\"\\s*:\\s*\"([^\"]*)\"");
            return match.Success ? match.Groups[1].Value : null;
        }

        private static int? ExtractYamlInt(string yaml, string key)
        {
            var match = System.Text.RegularExpressions.Regex.Match(
                yaml ?? string.Empty,
                @"(?m)^\s*" + System.Text.RegularExpressions.Regex.Escape(key) + @":\s*(-?\d+)\s*$");
            return match.Success && int.TryParse(match.Groups[1].Value, out var value) ? value : (int?)null;
        }
    }
}
