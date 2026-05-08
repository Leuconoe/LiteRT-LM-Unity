$LiteRtLmAndroidBenchmarks = @(
    [pscustomobject]@{
        Name = "gemma-4-E2B-it-gpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma4"
        Model = "gemma-4-E2B-it.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it.apk"
    },
    [pscustomobject]@{
        Name = "gemma-4-E2B-it-gpu-nospec"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma4NoSpeculative"
        Model = "gemma-4-E2B-it.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-nospec.apk"
    },
    [pscustomobject]@{
        Name = "gemma-4-E2B-it-cpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma4Cpu"
        Model = "gemma-4-E2B-it.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-CPU.apk"
    },
    [pscustomobject]@{
        Name = "gemma3-1b-it-gpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma1B"
        Model = "gemma3-1b-it-int4.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4.apk"
    },
    [pscustomobject]@{
        Name = "gemma3-1b-it-cpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma1BCpu"
        Model = "gemma3-1b-it-int4.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-CPU.apk"
    },
    [pscustomobject]@{
        Name = "gemma3-270m-it-gpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma270M"
        Model = "gemma3-270m-it-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8.apk"
    },
    [pscustomobject]@{
        Name = "mobile-actions-gpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkMobileActions"
        Model = "mobile_actions_q8_ekv1024.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-mobile_actions_q8_ekv1024.apk"
    },
    [pscustomobject]@{
        Name = "qwen3-0.6b-gpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen3"
        Model = "Qwen3-0.6B.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen3-0.6B.apk"
    },
    [pscustomobject]@{
        Name = "qwen2.5-0.5b-gpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25"
        Model = "Qwen2.5-0.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct.apk"
    },
    [pscustomobject]@{
        Name = "qwen2.5-0.5b-cpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25Cpu"
        Model = "Qwen2.5-0.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk"
    },
    [pscustomobject]@{
        Name = "qwen2.5-1.5b-gpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25_1_5B"
        Model = "Qwen2.5-1.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct.apk"
    },
    [pscustomobject]@{
        Name = "qwen2.5-1.5b-cpu"
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25_1_5BCpu"
        Model = "Qwen2.5-1.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct-CPU.apk"
    }
)

function Get-LiteRtLmAndroidBenchmark {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $benchmark = $LiteRtLmAndroidBenchmarks |
        Where-Object { [string]::Equals($_.Name, $Name, [StringComparison]::OrdinalIgnoreCase) } |
        Select-Object -First 1

    if ($null -eq $benchmark) {
        $availableNames = ($LiteRtLmAndroidBenchmarks | Select-Object -ExpandProperty Name) -join ", "
        throw "Unknown benchmark '$Name'. Available benchmarks: $availableNames"
    }

    return $benchmark
}

function Select-LiteRtLmAndroidBenchmarks {
    param([string[]]$Name = @())

    if ($Name.Count -eq 0) {
        return @($LiteRtLmAndroidBenchmarks)
    }

    $selected = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
    foreach ($benchmarkName in $Name) {
        [void]$selected.Add($benchmarkName)
    }

    $benchmarks = @($LiteRtLmAndroidBenchmarks | Where-Object { $selected.Contains($_.Name) })
    if ($benchmarks.Count -eq 0) {
        throw "No benchmarks matched -BenchmarkName: $($Name -join ', ')"
    }

    return $benchmarks
}
