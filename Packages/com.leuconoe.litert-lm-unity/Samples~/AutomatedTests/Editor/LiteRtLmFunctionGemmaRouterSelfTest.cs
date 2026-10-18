using System;
using System.Collections.Generic;
using System.Linq;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Self-test for <see cref="LiteRtLmFunctionGemmaRouter"/>: the asserts of fg_common.py's __main__ plus a
    /// golden table of 20 routes whose model outputs are the raw device responses of the fine-tuned
    /// FunctionGemma (v4e, c256-p64, device 46a880a0 — Builds/Logs/AndroidFcPromptSet/v4e-c256-p64) and whose
    /// expected results are what fg_common's host-side route produced from those same responses.
    /// Pure C# apart from the editor entry points, so it also runs outside Unity.
    /// </summary>
    public static class LiteRtLmFunctionGemmaRouterSelfTest
    {
        private const string Refused = @"{""arguments"":{},""function"":""DefaultResponse""}";
        private const string RawRefusal = @"<start_function_call>call:DefaultResponse{}<end_function_call>";

        private sealed class Golden
        {
            public string Utterance;
            public string Expected;
            public Dictionary<string, string> Raw;
        }

        private static Golden G(string utterance, string expected, params string[] textRawPairs)
        {
            var raw = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var i = 0; i < textRawPairs.Length; i += 2) raw[textRawPairs[i]] = textRawPairs[i + 1];
            return new Golden { Utterance = utterance, Expected = expected, Raw = raw };
        }

        // Device raw outputs (text → model response) and the Python route result for each utterance.
        private static readonly Golden[] Goldens =
        {
            // plain executes
            G(@"좋아, 다음은 유니폼팀", @"{""arguments"":{""team"":""Uniform""},""function"":""SelectTeam""}", @"좋아, 다음은 유니폼팀", @"<start_function_call>call:SelectTeam{team:<escape>Uniform<escape>}<end_function_call>"),
            G(@"예비 드론 많네, 팀 편성 화면 열어 봐", @"{""arguments"":{},""function"":""EnterTeamCreation""}", @"예비 드론 많네, 팀 편성 화면 열어 봐", @"<start_function_call>call:EnterTeamCreation{}<end_function_call>"),
            G(@"남쪽이 비었다, 정찰 드론 세 대로 팀 생성해, 서둘러", @"{""arguments"":{""memberCount"":3,""teamType"":""정찰""},""function"":""CreateTeam""}", @"남쪽이 비었다, 정찰 드론 세 대로 팀 생성해, 서둘러", @"<start_function_call>call:CreateTeam{memberCount:3,teamType:<escape>정찰<escape>}<end_function_call>"),
            G(@"보급로 끊어야 한다, 투하 드론 네 대로 팀 만들어 주십시오", @"{""arguments"":{""memberCount"":4,""teamType"":""공격""},""function"":""CreateTeam""}", @"보급로 끊어야 한다, 투하 드론 네 대로 팀 만들어 주십시오", @"<start_function_call>call:CreateTeam{memberCount:4,teamType:<escape>공격<escape>}<end_function_call>"),
            G(@"적 전차 접근 중, 자폭 드론 오 기로 팀 편성해, 지금", @"{""arguments"":{""memberCount"":5,""teamType"":""자폭""},""function"":""CreateTeam""}", @"적 전차 접근 중, 자폭 드론 오 기로 팀 편성해, 지금", @"<start_function_call>call:CreateTeam{memberCount:5,teamType:<escape>자폭<escape>}<end_function_call>"),
            G(@"화력이 모자라, 폭격 드론 일곱 대로 팀 꾸려 줘, 급하다", @"{""arguments"":{""memberCount"":7,""teamType"":""공격""},""function"":""CreateTeam""}", @"화력이 모자라, 폭격 드론 일곱 대로 팀 꾸려 줘, 급하다", @"<start_function_call>call:CreateTeam{memberCount:7,teamType:<escape>공격<escape>}<end_function_call>"),
            // refused whole, rescued by the clause split (two of them also exercise the guard inside a clause)
            G(@"좌측 능선에 움직임 있다, 오스카팀 선택, 빨리", @"{""arguments"":{""team"":""Oscar""},""function"":""SelectTeam""}", @"좌측 능선에 움직임 있다, 오스카팀 선택, 빨리", RawRefusal, @"좌측 능선에 움직임 있다", RawRefusal, @"오스카팀 선택", @"<start_function_call>call:SelectTeam{team:<escape>Oscar<escape>}<end_function_call>", @"빨리", RawRefusal),
            G(@"연료 경고 떴습니다, 폭스트롯팀 귀환시켜 주세요, 지금요", @"{""arguments"":{""team"":""Foxtrot""},""function"":""ReturnTeam""}", @"연료 경고 떴습니다, 폭스트롯팀 귀환시켜 주세요, 지금요", RawRefusal, @"연료 경고 떴습니다", @"<start_function_call>call:ApplyDisplayFilters{filterNames:<escape>연료 경고<escape>}<end_function_call>", @"폭스트롯팀 귀환시켜 주세요", @"<start_function_call>call:ReturnTeam{team:<escape>Foxtrot<escape>}<end_function_call>", @"지금요", RawRefusal),
            G(@"진입로 확보 완료, 엑스레이팀 전술 시작해, 오버", @"{""arguments"":{""team"":""X-ray""},""function"":""StartTeamTactics""}", @"진입로 확보 완료, 엑스레이팀 전술 시작해, 오버", RawRefusal, @"진입로 확보 완료", @"<start_function_call>call:AcceptConfirmWindow{confirmName:<escape>진입로 확보<escape>}<end_function_call>", @"엑스레이팀 전술 시작해", @"<start_function_call>call:StartTeamTactics{team:<escape>X-ray<escape>}<end_function_call>", @"오버", RawRefusal),
            G(@"엄폐물 뒤에 있는 거, 델타팀 삼 번 드론 선택, 빨리", @"{""arguments"":{""droneNumber"":3,""team"":""Delta""},""function"":""SelectDrone""}", @"엄폐물 뒤에 있는 거, 델타팀 삼 번 드론 선택, 빨리", RawRefusal, @"엄폐물 뒤에 있는 거", RawRefusal, @"델타팀 삼 번 드론 선택", @"<start_function_call>call:SelectDrone{droneNumber:3,team:<escape>Delta<escape>}<end_function_call>", @"빨리", RawRefusal),
            G(@"배터리 넉넉한 놈으로, 일 번 드론 선택해 줘", @"{""arguments"":{""droneNumber"":1},""function"":""SelectDrone""}", @"배터리 넉넉한 놈으로, 일 번 드론 선택해 줘", RawRefusal, @"배터리 넉넉한 놈으로", RawRefusal, @"일 번 드론 선택해 줘", @"<start_function_call>call:SelectDrone{droneNumber:1}<end_function_call>"),
            // the guard refuses a call the model did make: invented number, invented callsign, enum miss
            G(@"드론 카메라 띄워", Refused, @"드론 카메라 띄워", @"<start_function_call>call:ShowDroneCamera{droneNumber:3}<end_function_call>"),
            G(@"제어권 김상사한테 넘겨", Refused, @"제어권 김상사한테 넘겨", @"<start_function_call>call:AssignTeamControl{soldierName:<escape>김상사<escape>,team:<escape>Royal<escape>}<end_function_call>"),
            G(@"위협 표시하고 필터 창 닫아", Refused, @"위협 표시하고 필터 창 닫아", @"<start_function_call>call:CloseWindow{windowName:<escape>위협<escape>}<end_function_call>"),
            // the model refused and the split adds nothing (one clause's call is ungrounded)
            G(@"찰리팀 제어권 서하사한테 넘겨, 이제 그 친구가 맡는다", Refused, @"찰리팀 제어권 서하사한테 넘겨, 이제 그 친구가 맡는다", RawRefusal, @"찰리팀 제어권 서하사한테 넘겨", @"<start_function_call>call:AssignTeamControl{soldierName:<escape>서방선박<escape>,team:<escape>Charlie<escape>}<end_function_call>", @"이제 그 친구가 맡는다", RawRefusal),
            G(@"마커 쓰리 지호", Refused, @"마커 쓰리 지호", RawRefusal),
            // known wrong actions — the port must reproduce them too (they are model errors, not guard errors)
            G(@"지상 수색 들어간다, 유지브이 두 대로 팀 생성 부탁해", @"{""arguments"":{},""function"":""EnterTeamCreation""}", @"지상 수색 들어간다, 유지브이 두 대로 팀 생성 부탁해", RawRefusal, @"지상 수색 들어간다", RawRefusal, @"유지브이 두 대로 팀 생성 부탁해", @"<start_function_call>call:EnterTeamCreation{}<end_function_call>"),
            G(@"호텔팀 제어권 문중위에게 이양합니다, 인계 바랍니다", @"{""arguments"":{""soldierName"":""제어권"",""team"":""Hotel""},""function"":""AssignTeamControl""}", @"호텔팀 제어권 문중위에게 이양합니다, 인계 바랍니다", @"<start_function_call>call:AssignTeamControl{soldierName:<escape>제어권<escape>,team:<escape>Hotel<escape>}<end_function_call>"),
            // synthetic utterances built from device clause outputs (heldout4 has no natural two-call case):
            // two different actions stay refused; the same action twice executes once
            G(@"오스카팀 선택, 엑스레이팀 전술 시작해", Refused, @"오스카팀 선택, 엑스레이팀 전술 시작해", RawRefusal, @"오스카팀 선택", @"<start_function_call>call:SelectTeam{team:<escape>Oscar<escape>}<end_function_call>", @"엑스레이팀 전술 시작해", @"<start_function_call>call:StartTeamTactics{team:<escape>X-ray<escape>}<end_function_call>"),
            G(@"오스카팀 선택, 오스카팀 선택", @"{""arguments"":{""team"":""Oscar""},""function"":""SelectTeam""}", @"오스카팀 선택, 오스카팀 선택", RawRefusal, @"오스카팀 선택", @"<start_function_call>call:SelectTeam{team:<escape>Oscar<escape>}<end_function_call>"),
        };

        /// <summary>Runs every check against <paramref name="toolsJson"/> (functiongemma_tools.json); returns the failures.</summary>
        public static List<string> Run(string toolsJson)
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

            // fg_common.py __main__ — parsing
            Check(J(LiteRtLmFunctionGemmaRouter.ParseCall("<start_function_call>call:SelectDrone{droneNumber:3,team:<escape>Alpha<escape>}<end_function_call>"))
                  == @"{""arguments"":{""droneNumber"":3,""team"":""Alpha""},""function"":""SelectDrone""}", "parse SelectDrone");
            Check(J(LiteRtLmFunctionGemmaRouter.ParseCall("<start_function_call>call:StartRecording{}<end_function_call>"))
                  == @"{""arguments"":{},""function"":""StartRecording""}", "parse empty args");
            var filters = LiteRtLmFunctionGemmaRouter.ParseArgs("filterNames:<escape>위협, 기동성<escape>");
            Check(filters.Count == 1 && (string)filters["filterNames"] == "위협, 기동성", "parse escaped comma");
            Check(LiteRtLmFunctionGemmaRouter.ParseCall("no call here") == null, "parse prose → null");

            // validate (schema)
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("MoveToDrone", "droneNumber", 1L), schema)) == Refused, "unknown function");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("ClearDisplayFilters", "filterNames", "팀 연결선, 지저분"), schema)) == Refused, "unknown filter");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("SelectTeam", "team", "알파"), schema)) == Refused, "enum miss");
            Check(LiteRtLmFunctionGemmaRouter.Validate(C("SelectDrone", "droneNumber", 3L, "team", "Alpha"), schema).Function == "SelectDrone", "valid call kept");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(null, schema)) == Refused, "null → refusal");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(C("SelectDrone", "droneNumber", "3"), schema)) == Refused, "string number → refusal");

            // validate (grounding)
            var cam = C("ShowDroneCamera", "droneNumber", 3L);
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(cam, schema, "드론 카메라 띄워")) == Refused, "invented number");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(cam, schema, "삼번 드론 카메라")) == J(cam), "Sino number word");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(cam, schema, "3번 드론 카메라")) == J(cam), "digit");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(cam, schema, "드론 셋 카메라")) == J(cam), "native number word");
            var ctl = C("AssignTeamControl", "team", "Charlie", "soldierName", "서방선박");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(ctl, schema, "찰리팀 제어권 서하사한테 넘겨")) == Refused, "invented soldier name");
            var jul = C("SelectTeam", "team", "Juliet");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(jul, schema, "줄리에팀 선택")) == J(jul), "STT callsign 줄리에");
            Check(J(LiteRtLmFunctionGemmaRouter.Validate(jul, schema, "킬로팀 선택")) == Refused, "invented callsign");
            Check(LiteRtLmFunctionGemmaRouter.Validate(C("SelectTeam", "team", "X-ray"), schema, "Xray팀").Function == "SelectTeam", "English callsign");

            // segments / combine
            Check(LiteRtLmFunctionGemmaRouter.Segments("좌측에 움직임 있다, 오스카팀 선택, 빨리").SequenceEqual(new[] { "좌측에 움직임 있다", "오스카팀 선택", "빨리" }), "segments split");
            Check(LiteRtLmFunctionGemmaRouter.Segments("오스카팀 선택").Count == 0, "single clause → none");
            var sel = C("SelectTeam", "team", "Oscar");
            var refusal = LiteRtLmFunctionGemmaRouter.Refusal();
            Check(J(LiteRtLmFunctionGemmaRouter.CombineSegments(refusal, new[] { refusal, sel, refusal })) == J(sel), "one call among remarks");
            Check(J(LiteRtLmFunctionGemmaRouter.CombineSegments(refusal, new[] { sel, C("StartRecording") })) == Refused, "two actions stay refused");
            Check(J(LiteRtLmFunctionGemmaRouter.CombineSegments(refusal, new[] { sel, C("SelectTeam", "team", "Oscar") })) == J(sel), "same action twice");

            // schema shape
            Check(schema.Required["SelectDrone"].SetEquals(new[] { "droneNumber" }), "SelectDrone required");
            Check(((List<object>)schema.Properties["SelectDrone"]["team"]["enum"]).Count == 26, "26 callsigns");

            // golden routes from device outputs
            foreach (var g in Goldens)
            {
                string Infer(string text) => g.Raw.TryGetValue(text, out var raw) ? raw : throw new InvalidOperationException($"no recorded output for '{text}'");
                string got;
                try
                {
                    got = LiteRtLmFunctionGemmaRouter.Route(g.Utterance, Infer, schema).ToJson();
                }
                catch (Exception ex)
                {
                    got = "threw " + ex.Message;
                }

                Check(got == g.Expected, $"golden '{g.Utterance}': want {g.Expected} got {got}");
            }

            return failures;
        }

        public static int GoldenCount => Goldens.Length;

#if UNITY_EDITOR
        private const string StatusRelativePath = "Builds/Logs/LiteRtLmFunctionGemmaRouterSelfTest.status.txt";

        [UnityEditor.MenuItem("LiteRT-LM/Tests/FunctionGemma Router Self-Test")]
        public static void RunFromMenu()
        {
            var failures = Run(LoadToolsJson());
            if (failures.Count == 0) UnityEngine.Debug.Log($"FunctionGemma router self-test: SUCCESS ({GoldenCount} golden routes).");
            else UnityEngine.Debug.LogError("FunctionGemma router self-test FAILED:\n" + string.Join("\n", failures));
        }

        /// <summary>For Run-LiteRtLmEditorSelfTest.ps1 -ExecuteMethod … -StatusRelativePath Builds\Logs\LiteRtLmFunctionGemmaRouterSelfTest.status.txt</summary>
        public static void RunBatchmode()
        {
            var statusPath = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(UnityEngine.Application.dataPath) ?? ".", StatusRelativePath);
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(statusPath));
            var failures = Run(LoadToolsJson());
            var phase = failures.Count == 0 ? "SUCCESS" : "FAILURE";
            var message = failures.Count == 0 ? $"{GoldenCount} golden routes" : string.Join(" | ", failures);
            System.IO.File.WriteAllText(statusPath, $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}");
            UnityEditor.EditorApplication.Exit(failures.Count == 0 ? 0 : 1);
        }

        // The schema ships with the Test Scenes sample (AsrFunctionCalling/Resources/functiongemma_tools.json).
        private static string LoadToolsJson()
        {
            var asset = UnityEngine.Resources.Load<UnityEngine.TextAsset>("functiongemma_tools");
            if (asset == null) throw new InvalidOperationException("functiongemma_tools.json not found — import the Test Scenes sample.");
            return asset.text;
        }
#endif
    }
}
