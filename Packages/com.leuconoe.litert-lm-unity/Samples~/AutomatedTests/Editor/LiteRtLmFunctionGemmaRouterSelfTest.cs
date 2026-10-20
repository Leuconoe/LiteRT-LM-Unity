using System;
using System.Collections.Generic;
using System.Linq;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Self-test for <see cref="LiteRtLmFunctionGemmaRouter"/> against the sample contract: parsing, the guard,
    /// spoken numbers, the clause split and the confirmation policy, plus every route of
    /// <c>functiongemma_router_golden.json</c> (next to this file). The same table runs through the Python twin
    /// (<c>rebuild_check.py selftest</c>), so the two routers cannot drift apart unnoticed.
    /// Pure C# apart from the editor entry points, so it also runs outside Unity.
    /// </summary>
    public static class LiteRtLmFunctionGemmaRouterSelfTest
    {
        private const string Refused = @"{""arguments"":{},""function"":""DefaultResponse""}";

        /// <summary>Runs every check; returns the failures. <paramref name="goldenJson"/> = functiongemma_router_golden.json.</summary>
        public static List<string> Run(string toolsJson, string goldenJson)
        {
            var failures = new List<string>();
            void Check(bool ok, string what)
            {
                if (!ok) failures.Add(what);
            }

            var schema = LiteRtLmFunctionGemmaRouter.ToolSchema.FromOpenAiJson(toolsJson);
            string J(LiteRtLmFunctionGemmaRouter.Call c) => c?.ToJson();
            LiteRtLmFunctionGemmaRouter.Call C(string name, params object[] kv)
            {
                var call = new LiteRtLmFunctionGemmaRouter.Call { Function = name };
                for (var i = 0; i < kv.Length; i += 2) call.Arguments[(string)kv[i]] = kv[i + 1];
                return call;
            }

            // parsing
            Check(J(LiteRtLmFunctionGemmaRouter.ParseCall("<start_function_call>call:SetAlarm{hour:7,minute:30}<end_function_call>"))
                  == @"{""arguments"":{""hour"":7,""minute"":30},""function"":""SetAlarm""}", "parse SetAlarm");
            Check(J(LiteRtLmFunctionGemmaRouter.ParseCall("<start_function_call>call:PauseMusic{}<end_function_call>"))
                  == @"{""arguments"":{},""function"":""PauseMusic""}", "parse empty args");
            var escaped = LiteRtLmFunctionGemmaRouter.ParseArgs("room:<escape>a, b:c<escape>,n:3");
            Check(escaped.Count == 2 && (string)escaped["room"] == "a, b:c" && (long)escaped["n"] == 3, "parse escaped comma and colon");
            Check(LiteRtLmFunctionGemmaRouter.ParseCall("no call here") == null, "parse prose → null");

            // spoken numbers (fg_common.number_forms)
            Check(LiteRtLmFunctionGemmaRouter.NumberForms(3).OrderBy(s => s, StringComparer.Ordinal).SequenceEqual(new[] { "3", "삼", "세", "셋" }.OrderBy(s => s, StringComparer.Ordinal)), "forms of 3");
            Check(LiteRtLmFunctionGemmaRouter.NumberForms(24).Contains("이십사") && LiteRtLmFunctionGemmaRouter.NumberForms(20).Contains("스무")
                  && LiteRtLmFunctionGemmaRouter.NumberForms(12).Contains("열두") && LiteRtLmFunctionGemmaRouter.NumberForms(115).Contains("백십오"), "Sino / native / counter forms");

            // validate (schema)
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("TurnOnLight", "room", "Garage"), schema)) == Refused, "enum miss");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("TurnOnLight"), schema)) == Refused, "required missing");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("SetVolume", "level", "7"), schema)) == Refused, "string number");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("SetVolume", "level", 70L), schema)) == Refused, "above maximum");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("Nope"), schema)) == Refused, "unknown function");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(null, schema)) == Refused, "null → refusal");

            // validate (grounding)
            var light = C("TurnOnLight", "room", "Kitchen");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(light, schema, "부엌 불 켜 줘")) == J(light), "alias grounds the enum");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(light, schema, "kitchen 불")) == J(light), "the value itself grounds");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(light, schema, "거실 불 켜 줘")) == Refused, "another room is not grounded");
            var temp = C("SetTemperature", "degrees", 24L);
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(temp, schema, "온도 이십사 도")) == J(temp), "Sino number word");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(temp, schema, "온도 24도")) == J(temp), "digits");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(temp, schema, "온도 좀 올려")) == Refused, "invented number");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("SetAlarm", "hour", 7L), schema, "일곱 시에 깨워 줘")) == J(C("SetAlarm", "hour", 7L)), "native number");

            // segments / combine
            Check(LiteRtLmFunctionGemmaRouter.Segments("좀 춥네, 온도 이십오 도로 맞춰 줘").SequenceEqual(new[] { "좀 춥네", "온도 이십오 도로 맞춰 줘" }), "segments split");
            Check(LiteRtLmFunctionGemmaRouter.Segments("불 꺼").Count == 0, "single clause → none");
            var refusal = LiteRtLmFunctionGemmaRouter.Refusal();
            Check(J(LiteRtLmFunctionGemmaRouter.CombineSegments(refusal, new[] { refusal, light })) == J(light), "one call among remarks");
            Check(J(LiteRtLmFunctionGemmaRouter.CombineSegments(refusal, new[] { light, C("PauseMusic") })) == Refused, "two actions stay refused");

            // confirmation policy and answers
            var policy = new LiteRtLmFunctionGemmaRouter.ConfirmationPolicy();
            Check(policy.Requires(C("UnlockDoor"), schema), "x-confirm function needs confirmation");
            Check(!policy.Requires(light, schema) && !policy.Requires(refusal, schema), "ordinary call / refusal run");
            policy.Functions.Add("TurnOnLight");
            Check(policy.Requires(light, schema), "app-added function needs confirmation");
            Check(LiteRtLmFunctionGemmaRouter.ParseConfirmation("예") == true && LiteRtLmFunctionGemmaRouter.ParseConfirmation("네, 실행해") == true, "yes");
            Check(LiteRtLmFunctionGemmaRouter.ParseConfirmation("아니오") == false && LiteRtLmFunctionGemmaRouter.ParseConfirmation("취소") == false, "no");
            Check(LiteRtLmFunctionGemmaRouter.ParseConfirmation("거실 불 켜") == null, "unrelated answer → null");

            // the shared golden table
            var routes = (List<object>)((Dictionary<string, object>)LiteRtLmFunctionGemmaRouter.ParseJson(goldenJson))["routes"];
            foreach (Dictionary<string, object> g in routes)
            {
                var utterance = (string)g["utterance"];
                var raw = (Dictionary<string, object>)g["raw"];
                var expected = (Dictionary<string, object>)g["expected"];
                var want = new LiteRtLmFunctionGemmaRouter.Call { Function = (string)expected["function"], Arguments = (Dictionary<string, object>)expected["arguments"] };
                var routePolicy = new LiteRtLmFunctionGemmaRouter.ConfirmationPolicy
                {
                    MinAsrConfidence = g.TryGetValue("min_asr_confidence", out var min) ? Convert.ToDouble(min) : 0,
                };
                double? confidence = g.TryGetValue("asr_confidence", out var conf) ? Convert.ToDouble(conf) : (double?)null;
                var wantConfirm = g.TryGetValue("confirm", out var c) && (bool)c;
                string got;
                var gotConfirm = false;
                try
                {
                    var call = LiteRtLmFunctionGemmaRouter.Route(utterance,
                        text => raw.TryGetValue(text, out var r) ? (string)r : throw new InvalidOperationException($"no recorded output for '{text}'"),
                        schema, routePolicy, confidence);
                    got = call.ToJson();
                    gotConfirm = call.NeedsConfirmation;
                }
                catch (Exception ex)
                {
                    got = "threw " + ex.Message;
                }

                Check(got == want.ToJson() && gotConfirm == wantConfirm,
                      $"golden '{utterance}': want {want.ToJson()} confirm={wantConfirm}, got {got} confirm={gotConfirm}");
            }

            return failures;
        }

#if UNITY_EDITOR
        private const string StatusRelativePath = "Builds/Logs/LiteRtLmFunctionGemmaRouterSelfTest.status.txt";

        [UnityEditor.MenuItem("LiteRT-LM/Tests/FunctionGemma Router Self-Test")]
        public static void RunFromMenu()
        {
            var failures = Run(LoadToolsJson(), LoadGoldenJson());
            if (failures.Count == 0) UnityEngine.Debug.Log("FunctionGemma router self-test: SUCCESS.");
            else UnityEngine.Debug.LogError("FunctionGemma router self-test FAILED:\n" + string.Join("\n", failures));
        }

        /// <summary>For batchmode: -executeMethod LiteRTLM.Unity.Editor.LiteRtLmFunctionGemmaRouterSelfTest.RunBatchmode</summary>
        public static void RunBatchmode()
        {
            var statusPath = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(UnityEngine.Application.dataPath) ?? ".", StatusRelativePath);
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(statusPath));
            var failures = Run(LoadToolsJson(), LoadGoldenJson());
            var phase = failures.Count == 0 ? "SUCCESS" : "FAILURE";
            System.IO.File.WriteAllText(statusPath, $"[{DateTime.UtcNow:O}] {phase}: {(failures.Count == 0 ? "ok" : string.Join(" | ", failures))}{Environment.NewLine}");
            UnityEditor.EditorApplication.Exit(failures.Count == 0 ? 0 : 1);
        }

        // The contract ships with the Test Scenes sample (AsrFunctionCalling/Resources/functiongemma_tools.json).
        private static string LoadToolsJson()
        {
            var asset = UnityEngine.Resources.Load<UnityEngine.TextAsset>("functiongemma_tools");
            if (asset == null) throw new InvalidOperationException("functiongemma_tools.json not found — import the Test Scenes sample.");
            return asset.text;
        }

        private static string LoadGoldenJson()
        {
            var guid = UnityEditor.AssetDatabase.FindAssets("functiongemma_router_golden").FirstOrDefault();
            if (guid == null) throw new InvalidOperationException("functiongemma_router_golden.json not found — import the Automated Tests sample.");
            return System.IO.File.ReadAllText(UnityEditor.AssetDatabase.GUIDToAssetPath(guid));
        }
#endif
    }
}
