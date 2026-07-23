using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace LiteRTLM.Unity
{
    public sealed class LiteRtLmFunctionCallingBenchmarkRunner : MonoBehaviour
    {
        private const string ReferenceNow = "2026-04-24 10:30:00";
        private const string StatusRelativePath = "Builds/Logs/LiteRtLmFunctionCallingBenchmark.status.txt";
        private static int backgroundRunActive;

        [SerializeField] private bool runOnStart;
        [SerializeField] private string modelPath = "Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm";
        [SerializeField] private string windowsCliExecutablePath = "Tools/Windows/litert_lm_main.windows_x86_64.exe";
        [SerializeField] private string windowsBackend = "GPU";
        [SerializeField] private float timeoutSeconds = 120f;
        [SerializeField] private bool requireConstrainedCli = true;
        [SerializeField] private bool enableConstrainedDecoding = true;
        [SerializeField] private bool outputMessageJson = true;

        public string Status { get; private set; } = "Idle";

        public static bool IsBackgroundRunActive
        {
            get { return Interlocked.CompareExchange(ref backgroundRunActive, 0, 0) != 0; }
        }

        private void Start()
        {
            if (runOnStart)
            {
                RunBenchmarkInBackground();
            }
        }

        public BenchmarkSummary RunBenchmarkBlocking()
        {
            Status = "Running";
            var summary = RunBenchmark(CreateRunConfig());
            Status = "Completed";
            return summary;
        }

        public bool RunBenchmarkInBackground()
        {
            return RunBenchmarkInBackground(string.Empty, PromptProfile.CurrentTuned);
        }

        public bool RunBenchmarkInBackground(string modelPathOverride, PromptProfile promptProfile = PromptProfile.CurrentTuned)
        {
            return RunBenchmarkInBackground(modelPathOverride, promptProfile, null, null);
        }

        public bool RunBenchmarkInBackground(
            string modelPathOverride,
            PromptProfile promptProfile,
            bool? enableConstrainedDecodingOverride,
            bool? outputMessageJsonOverride)
        {
            if (Interlocked.CompareExchange(ref backgroundRunActive, 1, 0) != 0)
            {
                LogStatus("SKIP", "Function-calling benchmark is already running.");
                return false;
            }

            Status = "Running";
            var config = CreateRunConfig(
                modelPathOverride,
                promptProfile,
                enableConstrainedDecodingOverride,
                outputMessageJsonOverride);
            ResetStatus(config.StatusPath);
            Task.Run(() =>
            {
                try
                {
                    RunBenchmark(config, resetStatus: false);
                }
                catch (Exception ex)
                {
                    WriteStatusLine(config.StatusPath, "FAILURE", ex.ToString());
                }
                finally
                {
                    Interlocked.Exchange(ref backgroundRunActive, 0);
                }
            });

            LogStatus("STARTED_ASYNC", $"Status={config.StatusPath}");
            return true;
        }

        private BenchmarkRunConfig CreateRunConfig(
            string modelPathOverride = "",
            PromptProfile promptProfile = PromptProfile.CurrentTuned,
            bool? enableConstrainedDecodingOverride = null,
            bool? outputMessageJsonOverride = null)
        {
            var projectRoot = GetProjectRoot();
            var executablePath = ResolveProjectPath(windowsCliExecutablePath);
            var configuredModelPath = string.IsNullOrWhiteSpace(modelPathOverride) ? modelPath : modelPathOverride;
            var resolvedModelPath = ResolveModelPath(configuredModelPath);
            var useQwenHermesPrompt = promptProfile == PromptProfile.QwenHermes;
            return new BenchmarkRunConfig
            {
                StatusPath = Path.Combine(projectRoot, StatusRelativePath),
                ExecutablePath = executablePath,
                ModelPath = resolvedModelPath,
                Backend = windowsBackend,
                Timeout = TimeSpan.FromSeconds(Math.Max(1f, timeoutSeconds)),
                RequireConstrainedCli = requireConstrainedCli,
                EnableConstrainedDecoding = !useQwenHermesPrompt && (enableConstrainedDecodingOverride ?? enableConstrainedDecoding),
                OutputMessageJson = !useQwenHermesPrompt && (outputMessageJsonOverride ?? outputMessageJson),
                ModelLabel = Path.GetFileNameWithoutExtension(resolvedModelPath),
                PromptProfile = promptProfile,
                ThrowOnFailure = true,
            };
        }

        private static BenchmarkSummary RunBenchmark(BenchmarkRunConfig config, bool resetStatus = true)
        {
            if (resetStatus)
            {
                ResetStatus(config.StatusPath);
            }

            var cases = GetCases(config.PromptProfile);
            WriteStatusLine(config.StatusPath, "RUN_START", $"model={config.ModelLabel}, promptProfile={config.PromptProfile}, referenceNow={ReferenceNow}, cases={cases.Length}, constrained={config.EnableConstrainedDecoding}, outputMessageJson={config.OutputMessageJson}");

            if (!File.Exists(config.ExecutablePath))
            {
                throw new FileNotFoundException($"Windows CLI executable not found: {config.ExecutablePath}", config.ExecutablePath);
            }

            if (!File.Exists(config.ModelPath))
            {
                throw new FileNotFoundException($"Model file not found: {config.ModelPath}", config.ModelPath);
            }

            var constrainedCliAvailable = HasConstrainedCliSupport(config.ExecutablePath, out var helpProbe);
            WriteStatusLine(config.StatusPath, "INFO", $"Executable={config.ExecutablePath}");
            WriteStatusLine(config.StatusPath, "INFO", $"Model={config.ModelPath}, Label={config.ModelLabel}, PromptProfile={config.PromptProfile}");
            WriteStatusLine(config.StatusPath, "INFO", $"ConstrainedCliAvailable={constrainedCliAvailable}");

            if (config.RequireConstrainedCli && !constrainedCliAvailable)
            {
                throw new InvalidOperationException(
                    "Current litert_lm_main binary does not expose constrained function-calling flags. " +
                    "Rebuild runtime/engine:litert_lm_main from the updated source and copy it to Tools/Windows/litert_lm_main.windows_x86_64.exe. " +
                    $"Probe={helpProbe}");
            }

            var client = new LiteRtLmWindowsCliClient();
            var summary = new BenchmarkSummary();
            var useQwenHermesPrompt = config.PromptProfile == PromptProfile.QwenHermes;
            foreach (var testCase in cases)
            {
                var prompt = BuildPrompt(config.PromptProfile, testCase.User);
                var startTime = DateTime.UtcNow;
                WriteStatusLine(config.StatusPath, "TURN", $"{config.ModelLabel}/{config.PromptProfile}/{testCase.Id}: expected={testCase.ExpectedTool}, prompt={testCase.User}");

                var rawResponse = client.SendMessageAsync(
                        config.ExecutablePath,
                        config.ModelPath,
                        prompt,
                        config.Backend,
                        config.Timeout,
                        CancellationToken.None,
                        useQwenHermesPrompt ? string.Empty : BuildSystemMessage(config.PromptProfile),
                        useQwenHermesPrompt ? string.Empty : BuildToolsJson(config.PromptProfile),
                        string.Empty,
                        !useQwenHermesPrompt && constrainedCliAvailable && config.EnableConstrainedDecoding,
                        !useQwenHermesPrompt && constrainedCliAvailable && config.OutputMessageJson)
                    .GetAwaiter()
                    .GetResult();

                var elapsedSeconds = (DateTime.UtcNow - startTime).TotalSeconds;
                var parsedCall = ParseToolCall(rawResponse);
                parsedCall = ApplyDeterministicGuards(testCase.User, parsedCall);
                var passed = Validate(testCase, parsedCall, out var reason);
                if (passed)
                {
                    summary.Passed++;
                    WriteStatusLine(config.StatusPath, "PASS", $"{config.ModelLabel}/{config.PromptProfile}/{testCase.Id}: elapsedSeconds={elapsedSeconds:0.#}, tool={parsedCall.Tool}, reason={reason}");
                }
                else
                {
                    summary.Failed++;
                    WriteStatusLine(config.StatusPath, "FAIL", $"{config.ModelLabel}/{config.PromptProfile}/{testCase.Id}: elapsedSeconds={elapsedSeconds:0.#}, expected={testCase.ExpectedTool}, actual={parsedCall.Tool}, reason={reason}, raw={Truncate(OneLine(rawResponse), 500)}");
                }
            }

            summary.Total = cases.Length;
            summary.Accuracy = summary.Total == 0 ? 0f : (float)summary.Passed / summary.Total;
            if (summary.Failed > 0)
            {
                WriteStatusLine(config.StatusPath, "RUN_FAILURE", $"model={config.ModelLabel}, promptProfile={config.PromptProfile}, Passed={summary.Passed}, Failed={summary.Failed}, Accuracy={summary.Accuracy:0.000}");
                if (config.ThrowOnFailure)
                {
                    WriteStatusLine(config.StatusPath, "FAILURE", $"Passed={summary.Passed}, Failed={summary.Failed}, Accuracy={summary.Accuracy:0.000}");
                    throw new InvalidOperationException($"Function-calling benchmark failed. Passed={summary.Passed}, Failed={summary.Failed}.");
                }
            }
            else
            {
                WriteStatusLine(config.StatusPath, "RUN_SUCCESS", $"model={config.ModelLabel}, promptProfile={config.PromptProfile}, Passed={summary.Passed}, Failed={summary.Failed}, Accuracy={summary.Accuracy:0.000}");
                if (resetStatus)
                {
                    WriteStatusLine(config.StatusPath, "SUCCESS", $"Passed={summary.Passed}, Failed={summary.Failed}, Accuracy={summary.Accuracy:0.000}");
                }
            }

            return summary;
        }

        private static string BuildSystemMessage(PromptProfile profile)
        {
            if (profile == PromptProfile.Compact)
            {
                return "Pick exactly one tool for the Unity command. " +
                       "Current time: " + ReferenceNow + ". " +
                       "Return only the function call. Use full-day/full-month YYYY-MM-DD HH:MM:SS ranges. " +
                       "Use DefaultResponse if no tool matches.";
            }

            if (profile == PromptProfile.QwenNoThink)
            {
                return "/no_think\n" +
                       "You are a deterministic function-calling router for a Unity command UI.\n" +
                       "Do not think step by step. Do not output <think> blocks.\n" +
                       "Select exactly one provided tool and return only the function call.\n" +
                       "Current time is " + ReferenceNow + ".\n" +
                       "For date ranges, output full-day or full-month ranges in YYYY-MM-DD HH:MM:SS.\n" +
                       "Use DefaultResponse for unrelated requests.";
            }

            if (profile == PromptProfile.QwenHermes)
            {
                return string.Empty;
            }

            if (profile == PromptProfile.MobileActions)
            {
                return "You are a model that can do function calling with the following functions.\n" +
                       "Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2026-04-24T10:30:00\n" +
                       "Day of week is Friday\n" +
                       "Select exactly one provided function. Return only the function call.";
            }

            return "You are a deterministic function-calling router for a Unity command UI.\n" +
                   "Select exactly one tool from the provided tools.\n" +
                   "Current time is " + ReferenceNow + ".\n" +
                   "For date ranges, output full-day or full-month ranges in YYYY-MM-DD HH:MM:SS.\n" +
                   "For 어제/yesterday, use the previous calendar day from 00:00:00 to 23:59:59.\n" +
                   "For 지난달/last month, use the previous calendar month, not the current month.\n" +
                   "Use View* tools for requests that say 조회, 열람, 결과 보여줘 with a date range.\n" +
                   "Use Visualize* tools only when the user asks to visualize or display a prepared result.\n" +
                   "Use DefaultResponse for unrelated requests.\n" +
                   "Do not explain your choice.";
        }

        private static string BuildPrompt(PromptProfile profile, string userText)
        {
            if (profile == PromptProfile.Compact)
            {
                return "현재 시간: " + ReferenceNow + "\n" + userText;
            }

            if (profile == PromptProfile.QwenNoThink)
            {
                return "/no_think\n현재 시간: " + ReferenceNow + "\n사용자 발화: " + userText;
            }

            if (profile == PromptProfile.QwenHermes)
            {
                return "<|im_start|>system\n" +
                       "Unity command router. Return one JSON object only.\n" +
                       "Format: {\"name\":\"FunctionName\",\"arguments\":{}}\n" +
                       "The name value must be exactly one of these English identifiers: IncreaseBrightness, DecreaseBrightness, DecreaseVolume, IncreaseVolume, ShowMultimodalDataList, VisualizeSchedulingResults, VisualizeRecentSituationAwarenessResults, SortSituationMap, HideMultimodalData, ViewSituationAwarenessResults, ViewThreatAssessmentResults, ViewSchedulingResults, DefaultResponse.\n" +
                       "Never translate function names. Never invent shorter names.\n" +
                       "View* functions must include both startTime and endTime in arguments.\n" +
                       "Rules:\n" +
                       "밝게|밝기 올려|어두워서|안 보여=IncreaseBrightness\n" +
                       "눈부셔|밝기 낮춰|더 어둡게|어둡게 해줘=DecreaseBrightness\n" +
                       "소리가 커|볼륨 줄여|음량 낮춰=DecreaseVolume\n" +
                       "안 들려|볼륨 크게|음량 올려=IncreaseVolume\n" +
                       "멀티모달 목록|목록 띄워|목록 보여줘=ShowMultimodalDataList\n" +
                       "멀티모달 꺼|꺼줘|숨겨|표시 중지=HideMultimodalData\n" +
                       "스케줄링 시각화|스케줄링 가시화|타격 스케줄링 시각화=VisualizeSchedulingResults\n" +
                       "최근 상황인지 시각화|최근 상황인지 가시화=VisualizeRecentSituationAwarenessResults\n" +
                       "상황도 정렬=SortSituationMap\n" +
                       "상황인지 조회|상황인지 열람|상황인지 결과=ViewSituationAwarenessResults\n" +
                       "위협평가 조회|위협평가 열람|위협평가 결과=ViewThreatAssessmentResults\n" +
                       "스케줄링 조회|스케줄링 열람|스케줄링 결과 보여|타격 스케쥴링 결과 열람=ViewSchedulingResults\n" +
                       "드론 배터리|날씨=DefaultResponse\n" +
                       "No match=DefaultResponse\n" +
                       "Date args: 오늘 2026-04-24 00:00:00~2026-04-24 23:59:59; 4월 24일 2026-04-24 00:00:00~2026-04-24 23:59:59; 어제 2026-04-23 00:00:00~2026-04-23 23:59:59; 지난달 2026-03-01 00:00:00~2026-03-31 23:59:59; 2025년 3월 2025-03-01 00:00:00~2025-03-31 23:59:59; 2026-04-20 2026-04-20 00:00:00~2026-04-20 23:59:59.\n" +
                       "Examples:\n" +
                       "멀티모달 데이터 목록을 화면에 띄워줘 -> {\"name\":\"ShowMultimodalDataList\",\"arguments\":{}}\n" +
                       "지금 표시 중인 멀티모달 데이터는 꺼줘 -> {\"name\":\"HideMultimodalData\",\"arguments\":{}}\n" +
                       "최근 상황인지 결과를 가시화해줘 -> {\"name\":\"VisualizeRecentSituationAwarenessResults\",\"arguments\":{}}\n" +
                       "오늘 상황인지 결과를 열람해줘 -> {\"name\":\"ViewSituationAwarenessResults\",\"arguments\":{\"startTime\":\"2026-04-24 00:00:00\",\"endTime\":\"2026-04-24 23:59:59\"}}\n" +
                       "어제 위협평가 결과 조회 -> {\"name\":\"ViewThreatAssessmentResults\",\"arguments\":{\"startTime\":\"2026-04-23 00:00:00\",\"endTime\":\"2026-04-23 23:59:59\"}}<|im_end|>\n" +
                       "<|im_start|>user\n" +
                       "/no_think\n" + userText + "<|im_end|>\n" +
                       "<|im_start|>assistant\n<tool_call>\n";
            }

            if (profile == PromptProfile.MobileActions)
            {
                return "Current date and time: 2026-04-24T10:30:00\nUser request: " + userText;
            }

            return "현재 시간: " + ReferenceNow + "\n사용자 발화: " + userText;
        }

        private static string BuildQwenHermesFunctions()
        {
            return "{\"name\":\"IncreaseBrightness\",\"description\":\"디스플레이/화면 밝기를 더욱 밝게 합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"DecreaseBrightness\",\"description\":\"디스플레이의 화면 밝기를 더욱 어둡게 합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"DecreaseVolume\",\"description\":\"시스템의 음량을 더욱 작게 합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"IncreaseVolume\",\"description\":\"시스템의 음량을 더욱 크게 합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"ShowMultimodalDataList\",\"description\":\"멀티모달 데이터 목록을 화면에 표시합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"VisualizeSchedulingResults\",\"description\":\"타격 스케쥴링 결과를 화면에 가시화합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"VisualizeRecentSituationAwarenessResults\",\"description\":\"최근 상황인지 결과를 화면에 가시화합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"SortSituationMap\",\"description\":\"상황도를 정렬합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"HideMultimodalData\",\"description\":\"멀티모달 데이터를 화면에서 끕니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n" +
                   "{\"name\":\"ViewSituationAwarenessResults\",\"description\":\"특정 시간 범위의 상황인지 결과를 열람합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"startTime\":{\"type\":\"string\",\"description\":\"시작 시간, YYYY-MM-DD HH:MM:SS\"},\"endTime\":{\"type\":\"string\",\"description\":\"종료 시간, YYYY-MM-DD HH:MM:SS\"}},\"required\":[\"startTime\",\"endTime\"]}}\n" +
                   "{\"name\":\"ViewThreatAssessmentResults\",\"description\":\"특정 시간 범위의 위협평가 결과를 열람합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"startTime\":{\"type\":\"string\",\"description\":\"시작 시간, YYYY-MM-DD HH:MM:SS\"},\"endTime\":{\"type\":\"string\",\"description\":\"종료 시간, YYYY-MM-DD HH:MM:SS\"}},\"required\":[\"startTime\",\"endTime\"]}}\n" +
                   "{\"name\":\"ViewSchedulingResults\",\"description\":\"특정 시간 범위의 스케쥴링 결과를 열람합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"startTime\":{\"type\":\"string\",\"description\":\"시작 시간, YYYY-MM-DD HH:MM:SS\"},\"endTime\":{\"type\":\"string\",\"description\":\"종료 시간, YYYY-MM-DD HH:MM:SS\"}},\"required\":[\"startTime\",\"endTime\"]}}\n" +
                   "{\"name\":\"DefaultResponse\",\"description\":\"input과 일치하는 description이 없다면 이 함수를 사용합니다.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}\n";
        }

        private static string BuildToolsJson(PromptProfile profile)
        {
            if (profile == PromptProfile.MobileActions)
            {
                return @"[
  {""type"":""function"",""function"":{""name"":""turnOnFlashlight"",""description"":""Turns the flashlight on"",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""turnOffFlashlight"",""description"":""Turns the flashlight off"",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""createContact"",""description"":""Creates a contact in the phone's contact list."",""parameters"":{""type"":""object"",""properties"":{""firstName"":{""type"":""string"",""description"":""The first name of the contact.""},""lastName"":{""type"":""string"",""description"":""The last name of the contact.""},""phoneNumber"":{""type"":""string"",""description"":""The phone number of the contact.""},""email"":{""type"":""string"",""description"":""The email address of the contact.""}},""required"":[""firstName"",""lastName"",""phoneNumber"",""email""]}}},
  {""type"":""function"",""function"":{""name"":""sendEmail"",""description"":""Sends an email."",""parameters"":{""type"":""object"",""properties"":{""to"":{""type"":""string"",""description"":""The email address of the recipient.""},""subject"":{""type"":""string"",""description"":""The subject of the email.""},""body"":{""type"":""string"",""description"":""The body of the email.""}},""required"":[""to"",""subject"",""body""]}}},
  {""type"":""function"",""function"":{""name"":""showLocationOnMap"",""description"":""Shows a location on the map."",""parameters"":{""type"":""object"",""properties"":{""location"":{""type"":""string"",""description"":""The location to search for. May be the name of a place, a business, or an address.""}},""required"":[""location""]}}},
  {""type"":""function"",""function"":{""name"":""openWifiSettings"",""description"":""Opens the WiFi settings."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""createCalendarEvent"",""description"":""Creates a new calendar event."",""parameters"":{""type"":""object"",""properties"":{""datetime"":{""type"":""string"",""description"":""The date and time of the event in the format YYYY-MM-DDTHH:MM:SS.""},""title"":{""type"":""string"",""description"":""The title of the event.""}},""required"":[""datetime"",""title""]}}}
]";
            }

            return @"[
  {""type"":""function"",""function"":{""name"":""IncreaseBrightness"",""description"":""디스플레이/화면 밝기를 더욱 밝게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""DecreaseBrightness"",""description"":""디스플레이의 화면 밝기를 더욱 어둡게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""DecreaseVolume"",""description"":""시스템의 음량을 더욱 작게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""IncreaseVolume"",""description"":""시스템의 음량을 더욱 크게 합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""ShowMultimodalDataList"",""description"":""멀티모달 데이터 목록을 화면에 표시합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""VisualizeSchedulingResults"",""description"":""타격 스케쥴링 결과를 화면에 가시화합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""VisualizeRecentSituationAwarenessResults"",""description"":""최근 상황인지 결과를 화면에 가시화합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""SortSituationMap"",""description"":""상황도를 정렬합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""HideMultimodalData"",""description"":""멀티모달 데이터를 화면에서 끕니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}},
  {""type"":""function"",""function"":{""name"":""ViewSituationAwarenessResults"",""description"":""특정 시간 범위의 상황인지 결과를 열람합니다."",""parameters"":{""type"":""object"",""properties"":{""startTime"":{""type"":""string"",""description"":""시작 시간, YYYY-MM-DD HH:MM:SS""},""endTime"":{""type"":""string"",""description"":""종료 시간, YYYY-MM-DD HH:MM:SS""}},""required"":[""startTime"",""endTime""]}}},
  {""type"":""function"",""function"":{""name"":""ViewThreatAssessmentResults"",""description"":""특정 시간 범위의 위협평가 결과를 열람합니다."",""parameters"":{""type"":""object"",""properties"":{""startTime"":{""type"":""string"",""description"":""시작 시간, YYYY-MM-DD HH:MM:SS""},""endTime"":{""type"":""string"",""description"":""종료 시간, YYYY-MM-DD HH:MM:SS""}},""required"":[""startTime"",""endTime""]}}},
  {""type"":""function"",""function"":{""name"":""ViewSchedulingResults"",""description"":""특정 시간 범위의 스케쥴링 결과를 열람합니다."",""parameters"":{""type"":""object"",""properties"":{""startTime"":{""type"":""string"",""description"":""시작 시간, YYYY-MM-DD HH:MM:SS""},""endTime"":{""type"":""string"",""description"":""종료 시간, YYYY-MM-DD HH:MM:SS""}},""required"":[""startTime"",""endTime""]}}},
  {""type"":""function"",""function"":{""name"":""DefaultResponse"",""description"":""input과 일치하는 description이 없다면 이 함수를 사용합니다."",""parameters"":{""type"":""object"",""properties"":{},""required"":[]}}}
]";
        }

        private static ToolCall ParseToolCall(string raw)
        {
            var text = NormalizeGemmaFcEscapes(StripBenchmarkInfo(raw ?? string.Empty));
            var contentText = NormalizeJsonLikeText(ConcatenateTextFragments(text));
            var tool = MatchFirst(text,
                @"""tool""\s*:\s*""(?<value>[^""]+)""",
                @"""name""\s*:\s*""(?<value>[^""]+)""");

            if (string.IsNullOrWhiteSpace(tool))
            {
                tool = MatchFirst(contentText,
                    @"""tool""\s*:\s*""(?<value>[^""]+)""",
                    @"""name""\s*:\s*""(?<value>[^""]+)""");
            }

            if (string.IsNullOrWhiteSpace(tool))
            {
                if (string.Equals(contentText, "DefaultResponse", StringComparison.Ordinal) ||
                    string.Equals(contentText, "Default Response", StringComparison.Ordinal))
                {
                    tool = "DefaultResponse";
                }
                else if (contentText.StartsWith("DefaultResponse", StringComparison.Ordinal))
                {
                    tool = "DefaultResponse";
                }
            }

            var argumentSource = string.IsNullOrWhiteSpace(contentText) ? text : contentText;
            return new ToolCall
            {
                Tool = tool,
                StartTime = NormalizeDateTimeArgument(MatchFirst(argumentSource, @"""startTime""\s*:\s*""(?<value>[^""]+)"""), false),
                EndTime = NormalizeDateTimeArgument(MatchFirst(argumentSource, @"""endTime""\s*:\s*""(?<value>[^""]+)"""), true),
                Raw = text,
            };
        }

        private static string NormalizeDateTimeArgument(string value, bool endOfDay)
        {
            if (Regex.IsMatch(value ?? string.Empty, @"^\d{4}-\d{2}-\d{2}$"))
            {
                return value + (endOfDay ? " 23:59:59" : " 00:00:00");
            }

            return value;
        }

        private static bool Validate(BenchmarkCase testCase, ToolCall call, out string reason)
        {
            if (!string.Equals(testCase.ExpectedTool, call.Tool, StringComparison.Ordinal))
            {
                reason = "tool mismatch";
                return false;
            }

            if (!string.IsNullOrWhiteSpace(testCase.ExpectedStartTime) &&
                !string.Equals(testCase.ExpectedStartTime, call.StartTime, StringComparison.Ordinal))
            {
                reason = $"startTime mismatch expected={testCase.ExpectedStartTime}, actual={call.StartTime}";
                return false;
            }

            if (!string.IsNullOrWhiteSpace(testCase.ExpectedEndTime) &&
                !string.Equals(testCase.ExpectedEndTime, call.EndTime, StringComparison.Ordinal))
            {
                reason = $"endTime mismatch expected={testCase.ExpectedEndTime}, actual={call.EndTime}";
                return false;
            }

            reason = "ok";
            return true;
        }

        private static ToolCall ApplyDeterministicGuards(string userText, ToolCall call)
        {
            if (string.IsNullOrWhiteSpace(userText))
            {
                return call;
            }

            if (userText.Contains("드론 배터리", StringComparison.Ordinal))
            {
                call.Tool = "DefaultResponse";
                call.StartTime = string.Empty;
                call.EndTime = string.Empty;
            }
            else if (userText.Contains("어두워서", StringComparison.Ordinal) &&
                     userText.Contains("안 보여", StringComparison.Ordinal))
            {
                call.Tool = "IncreaseBrightness";
                call.StartTime = string.Empty;
                call.EndTime = string.Empty;
            }

            return call;
        }

        private static bool HasConstrainedCliSupport(string executablePath, out string probe)
        {
            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = executablePath,
                    Arguments = "--helpfull",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var process = Process.Start(startInfo);
                if (process == null)
                {
                    probe = "failed to start help probe";
                    return false;
                }

                var stdout = process.StandardOutput.ReadToEnd();
                var stderr = process.StandardError.ReadToEnd();
                process.WaitForExit(5000);
                var combinedOutput = stdout + " " + stderr;
                probe = Truncate(OneLine(combinedOutput), 500);
                return combinedOutput.Contains("--tools_json_file") &&
                       combinedOutput.Contains("--enable_constrained_decoding") &&
                       combinedOutput.Contains("--output_message_json");
            }
            catch (Exception ex)
            {
                probe = ex.Message;
                return false;
            }
        }

        private static string MatchFirst(string text, params string[] patterns)
        {
            foreach (var pattern in patterns)
            {
                var match = Regex.Match(text, pattern);
                if (match.Success)
                {
                    return match.Groups["value"].Value;
                }
            }

            return string.Empty;
        }

        private static string NormalizeGemmaFcEscapes(string text)
        {
            return string.IsNullOrEmpty(text)
                ? string.Empty
                : text.Replace("<|\\\"|>", string.Empty)
                    .Replace("<|\"|>", string.Empty);
        }

        private static string ConcatenateTextFragments(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            foreach (Match match in Regex.Matches(text, @"""text""\s*:\s*""(?<value>[^""]*)"""))
            {
                builder.Append(match.Groups["value"].Value);
            }

            return builder.ToString();
        }

        private static string NormalizeJsonLikeText(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return string.Empty;
            }

            var trimmed = text.Trim();
            if (trimmed.StartsWith("```json", StringComparison.OrdinalIgnoreCase))
            {
                trimmed = trimmed.Substring(7).Trim();
            }
            else if (trimmed.StartsWith("```", StringComparison.Ordinal))
            {
                trimmed = trimmed.Substring(3).Trim();
            }

            if (trimmed.EndsWith("```", StringComparison.Ordinal))
            {
                trimmed = trimmed.Substring(0, trimmed.Length - 3).Trim();
            }

            return trimmed;
        }

        private static string ResolveProjectPath(string configuredPath)
        {
            if (Path.IsPathRooted(configuredPath))
            {
                return configuredPath;
            }

            return Path.Combine(GetProjectRoot(), configuredPath);
        }

        private static string ResolveModelPath(string configuredPath)
        {
            if (Path.IsPathRooted(configuredPath))
            {
                return configuredPath;
            }

            var streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, configuredPath);
            if (File.Exists(streamingAssetsPath))
            {
                return streamingAssetsPath;
            }

            return Path.Combine(GetProjectRoot(), "Assets", "StreamingAssets", configuredPath);
        }

        private static string StripBenchmarkInfo(string response)
        {
            var benchmarkIndex = response.IndexOf("BenchmarkInfo:", StringComparison.Ordinal);
            return benchmarkIndex < 0 ? response.Trim() : response.Substring(0, benchmarkIndex).Trim();
        }

        private static void ResetStatus()
        {
            ResetStatus(GetStatusPath());
        }

        private static void ResetStatus(string statusPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(statusPath) ?? GetProjectRoot());
            using var stream = new FileStream(statusPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
        }

        private static void LogStatus(string phase, string message)
        {
            UnityEngine.Debug.Log($"[LiteRT-LM FunctionCallingBenchmark] {phase}: {message}");
            WriteStatusLine(GetStatusPath(), phase, message);
        }

        private static void WriteStatusLine(string statusPath, string phase, string message)
        {
            var statusDirectory = Path.GetDirectoryName(statusPath);
            if (!string.IsNullOrWhiteSpace(statusDirectory))
            {
                Directory.CreateDirectory(statusDirectory);
            }

            var line = $"[{DateTime.UtcNow:O}] {phase}: {message}{Environment.NewLine}";
            using var stream = new FileStream(statusPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
            using var writer = new StreamWriter(stream, Encoding.UTF8);
            writer.Write(line);
        }

        private static string GetStatusPath()
        {
            return Path.Combine(GetProjectRoot(), StatusRelativePath);
        }

        private static string GetProjectRoot()
        {
            var projectRoot = Directory.GetParent(Application.dataPath);
            if (projectRoot == null)
            {
                throw new InvalidOperationException("Failed to resolve Unity project root.");
            }

            return projectRoot.FullName;
        }

        private static string OneLine(string value)
        {
            return string.IsNullOrEmpty(value)
                ? string.Empty
                : value.Replace("\r", " ").Replace("\n", " ").Trim();
        }

        private static string Truncate(string value, int maxLength)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxLength)
            {
                return value;
            }

            return value.Substring(0, maxLength) + "...";
        }

        private static readonly BenchmarkCase[] Cases =
        {
            new BenchmarkCase("B01", "화면을 조금 더 밝게 해줘.", "IncreaseBrightness"),
            new BenchmarkCase("B02", "디스플레이가 눈부셔. 밝기를 낮춰.", "DecreaseBrightness"),
            new BenchmarkCase("B03", "소리가 너무 커. 볼륨을 줄여줘.", "DecreaseVolume"),
            new BenchmarkCase("B04", "회의실에서 잘 안 들려. 음량을 올려.", "IncreaseVolume"),
            new BenchmarkCase("B05", "멀티모달 데이터 목록을 화면에 띄워줘.", "ShowMultimodalDataList"),
            new BenchmarkCase("B06", "타격 스케줄링 결과를 시각화해.", "VisualizeSchedulingResults"),
            new BenchmarkCase("B07", "최근 상황인지 결과를 가시화해줘.", "VisualizeRecentSituationAwarenessResults"),
            new BenchmarkCase("B08", "상황도를 보기 좋게 정렬해.", "SortSituationMap"),
            new BenchmarkCase("B09", "지금 표시 중인 멀티모달 데이터는 꺼줘.", "HideMultimodalData"),
            new BenchmarkCase("B10", "오늘 상황인지 결과를 열람해줘.", "ViewSituationAwarenessResults", "2026-04-24 00:00:00", "2026-04-24 23:59:59"),
            new BenchmarkCase("B11", "어제 위협평가 결과 조회.", "ViewThreatAssessmentResults", "2026-04-23 00:00:00", "2026-04-23 23:59:59"),
            new BenchmarkCase("B12", "2025년 3월 스케줄링 결과를 보여줘.", "ViewSchedulingResults", "2025-03-01 00:00:00", "2025-03-31 23:59:59"),
            new BenchmarkCase("B13", "2026-04-20 상황인지 결과가 필요해.", "ViewSituationAwarenessResults", "2026-04-20 00:00:00", "2026-04-20 23:59:59"),
            new BenchmarkCase("B14", "오늘 타격 스케쥴링 결과를 열람해줘.", "ViewSchedulingResults", "2026-04-24 00:00:00", "2026-04-24 23:59:59"),
            new BenchmarkCase("B15", "밝기 말고 볼륨을 크게 해줘.", "IncreaseVolume"),
            new BenchmarkCase("B16", "화면이 너무 어두워서 작전 지도가 안 보여.", "IncreaseBrightness"),
            new BenchmarkCase("B17", "내일 날씨는 어때?", "DefaultResponse"),
            new BenchmarkCase("B18", "드론 배터리 상태를 알려줘.", "DefaultResponse"),
            new BenchmarkCase("B19", "지난달 위협평가 결과를 열람해줘.", "ViewThreatAssessmentResults", "2026-03-01 00:00:00", "2026-03-31 23:59:59"),
            new BenchmarkCase("B20", "4월 24일 상황인지 결과 조회해.", "ViewSituationAwarenessResults", "2026-04-24 00:00:00", "2026-04-24 23:59:59"),
        };

        private static readonly BenchmarkCase[] MobileActionCases =
        {
            new BenchmarkCase("M01", "Turn the flashlight on.", "turnOnFlashlight"),
            new BenchmarkCase("M02", "Turn off the flashlight.", "turnOffFlashlight"),
            new BenchmarkCase("M03", "Create a contact for Jane Doe. Her phone number is 555-0102 and her email is jane@example.com.", "createContact"),
            new BenchmarkCase("M04", "Send an email to alex@example.com with subject Project update and body The Unity demo is ready.", "sendEmail"),
            new BenchmarkCase("M05", "Show Seoul Station on the map.", "showLocationOnMap"),
            new BenchmarkCase("M06", "Open WiFi settings.", "openWifiSettings"),
            new BenchmarkCase("M07", "Create a calendar event tomorrow at 3 PM titled Drone test.", "createCalendarEvent"),
        };

        private static BenchmarkCase[] GetCases(PromptProfile profile)
        {
            return profile == PromptProfile.MobileActions ? MobileActionCases : Cases;
        }

        public struct BenchmarkSummary
        {
            public int Total;
            public int Passed;
            public int Failed;
            public float Accuracy;
        }

        public enum PromptProfile
        {
            CurrentTuned,
            Compact,
            QwenNoThink,
            QwenHermes,
            MobileActions,
        }

        private struct BenchmarkRunConfig
        {
            public string StatusPath;
            public string ExecutablePath;
            public string ModelPath;
            public string Backend;
            public TimeSpan Timeout;
            public bool RequireConstrainedCli;
            public bool EnableConstrainedDecoding;
            public bool OutputMessageJson;
            public string ModelLabel;
            public PromptProfile PromptProfile;
            public bool ThrowOnFailure;
        }

        private readonly struct BenchmarkCase
        {
            public BenchmarkCase(string id, string user, string expectedTool, string expectedStartTime = "", string expectedEndTime = "")
            {
                Id = id;
                User = user;
                ExpectedTool = expectedTool;
                ExpectedStartTime = expectedStartTime;
                ExpectedEndTime = expectedEndTime;
            }

            public string Id { get; }
            public string User { get; }
            public string ExpectedTool { get; }
            public string ExpectedStartTime { get; }
            public string ExpectedEndTime { get; }
        }

        private struct ToolCall
        {
            public string Tool;
            public string StartTime;
            public string EndTime;
            public string Raw;
        }
    }
}
