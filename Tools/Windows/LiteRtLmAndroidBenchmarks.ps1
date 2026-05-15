function New-LiteRtLmAndroidBenchmark {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$DisplayName,
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$Model,
        [Parameter(Mandatory = $true)]
        [string]$Backend,
        [Parameter(Mandatory = $true)]
        [string]$Apk,
        [bool]$Speculative = $false,
        [int]$MaxNumTokens = 64,
        [int]$MaxNumImages = 0,
        [int]$BenchmarkPrefillTokens = 64,
        [int]$PreferredRank = 999,
        [string]$Notes = ""
    )

    [pscustomobject]@{
        Name = $Name
        DisplayName = $DisplayName
        Repo = $Repo
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkFromCommandLine"
        Model = $Model
        Backend = $Backend
        Apk = $Apk
        Speculative = $Speculative
        MaxNumTokens = $MaxNumTokens
        MaxNumImages = $MaxNumImages
        BenchmarkPrefillTokens = $BenchmarkPrefillTokens
        PreferredRank = $PreferredRank
        Notes = $Notes
    }
}

$LiteRtLmAndroidBenchmarks = @(
    New-LiteRtLmAndroidBenchmark -Name "gemma-4-e2b-it-gpu" -DisplayName "Gemma 4 E2B IT" -Repo "litert-community/gemma-4-E2B-it-litert-lm" -Model "gemma-4-E2B-it.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-GPU.apk" -Speculative $true -MaxNumTokens 4000 -BenchmarkPrefillTokens 128 -PreferredRank 1 -Notes "Primary recommended model."
    New-LiteRtLmAndroidBenchmark -Name "gemma-4-e2b-it-cpu" -DisplayName "Gemma 4 E2B IT" -Repo "litert-community/gemma-4-E2B-it-litert-lm" -Model "gemma-4-E2B-it.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-CPU.apk" -Speculative $true -MaxNumTokens 4000 -BenchmarkPrefillTokens 128 -PreferredRank 1 -Notes "CPU comparison for the primary recommended model."

    New-LiteRtLmAndroidBenchmark -Name "gemma-4-e4b-it-gpu" -DisplayName "Gemma 4 E4B IT" -Repo "litert-community/gemma-4-E4B-it-litert-lm" -Model "gemma-4-E4B-it.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-4-E4B-it-GPU.apk" -Speculative $true -MaxNumTokens 4000 -BenchmarkPrefillTokens 128
    New-LiteRtLmAndroidBenchmark -Name "gemma-4-e4b-it-cpu" -DisplayName "Gemma 4 E4B IT" -Repo "litert-community/gemma-4-E4B-it-litert-lm" -Model "gemma-4-E4B-it.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-4-E4B-it-CPU.apk" -Speculative $true -MaxNumTokens 4000 -BenchmarkPrefillTokens 128

    New-LiteRtLmAndroidBenchmark -Name "gemma-3-270m-it-gpu" -DisplayName "Gemma 3 270M IT" -Repo "litert-community/gemma-3-270m-it" -Model "gemma3-270m-it-q8.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8-GPU.apk"
    New-LiteRtLmAndroidBenchmark -Name "gemma-3-270m-it-cpu" -DisplayName "Gemma 3 270M IT" -Repo "litert-community/gemma-3-270m-it" -Model "gemma3-270m-it-q8.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8-CPU.apk"

    New-LiteRtLmAndroidBenchmark -Name "gemma-3n-e2b-it-gpu" -DisplayName "Gemma 3n E2B IT" -Repo "google/gemma-3n-E2B-it-litert-lm" -Model "gemma-3n-E2B-it-int4.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-3n-E2B-it-GPU.apk" -MaxNumTokens 4000 -BenchmarkPrefillTokens 128
    New-LiteRtLmAndroidBenchmark -Name "gemma-3n-e2b-it-cpu" -DisplayName "Gemma 3n E2B IT" -Repo "google/gemma-3n-E2B-it-litert-lm" -Model "gemma-3n-E2B-it-int4.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-3n-E2B-it-CPU.apk" -MaxNumTokens 4000 -BenchmarkPrefillTokens 128

    New-LiteRtLmAndroidBenchmark -Name "gemma-3n-e4b-it-gpu" -DisplayName "Gemma 3n E4B IT" -Repo "google/gemma-3n-E4B-it-litert-lm" -Model "gemma-3n-E4B-it-int4.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-3n-E4B-it-GPU.apk" -MaxNumTokens 4000 -BenchmarkPrefillTokens 128
    New-LiteRtLmAndroidBenchmark -Name "gemma-3n-e4b-it-cpu" -DisplayName "Gemma 3n E4B IT" -Repo "google/gemma-3n-E4B-it-litert-lm" -Model "gemma-3n-E4B-it-int4.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-gemma-3n-E4B-it-CPU.apk" -MaxNumTokens 4000 -BenchmarkPrefillTokens 128

    New-LiteRtLmAndroidBenchmark -Name "gemma3-1b-it-gpu" -DisplayName "Gemma 3 1B IT" -Repo "litert-community/Gemma3-1B-IT" -Model "gemma3-1b-it-int4.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-GPU.apk" -PreferredRank 3
    New-LiteRtLmAndroidBenchmark -Name "gemma3-1b-it-cpu" -DisplayName "Gemma 3 1B IT" -Repo "litert-community/Gemma3-1B-IT" -Model "gemma3-1b-it-int4.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-CPU.apk" -PreferredRank 3

    New-LiteRtLmAndroidBenchmark -Name "phi-4-mini-instruct-gpu" -DisplayName "Phi-4 Mini Instruct" -Repo "litert-community/Phi-4-mini-instruct" -Model "Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-phi-4-mini-instruct-GPU.apk"
    New-LiteRtLmAndroidBenchmark -Name "phi-4-mini-instruct-cpu" -DisplayName "Phi-4 Mini Instruct" -Repo "litert-community/Phi-4-mini-instruct" -Model "Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-phi-4-mini-instruct-CPU.apk"

    New-LiteRtLmAndroidBenchmark -Name "qwen2.5-1.5b-gpu" -DisplayName "Qwen2.5 1.5B Instruct" -Repo "litert-community/Qwen2.5-1.5B-Instruct" -Model "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct-GPU.apk"
    New-LiteRtLmAndroidBenchmark -Name "qwen2.5-1.5b-cpu" -DisplayName "Qwen2.5 1.5B Instruct" -Repo "litert-community/Qwen2.5-1.5B-Instruct" -Model "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct-CPU.apk"

    New-LiteRtLmAndroidBenchmark -Name "deepseek-r1-distill-qwen-1.5b-gpu" -DisplayName "DeepSeek R1 Distill Qwen 1.5B" -Repo "litert-community/DeepSeek-R1-Distill-Qwen-1.5B" -Model "DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-deepseek-r1-distill-qwen-1.5b-GPU.apk"
    New-LiteRtLmAndroidBenchmark -Name "deepseek-r1-distill-qwen-1.5b-cpu" -DisplayName "DeepSeek R1 Distill Qwen 1.5B" -Repo "litert-community/DeepSeek-R1-Distill-Qwen-1.5B" -Model "DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-deepseek-r1-distill-qwen-1.5b-CPU.apk"

    New-LiteRtLmAndroidBenchmark -Name "smollm-135m-instruct-gpu" -DisplayName "SmolLM 135M Instruct" -Repo "litert-community/SmolLM-135M-Instruct" -Model "SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-smollm-135m-instruct-GPU.apk"
    New-LiteRtLmAndroidBenchmark -Name "smollm-135m-instruct-cpu" -DisplayName "SmolLM 135M Instruct" -Repo "litert-community/SmolLM-135M-Instruct" -Model "SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-smollm-135m-instruct-CPU.apk"

    New-LiteRtLmAndroidBenchmark -Name "tinyllama-1.1b-chat-gpu" -DisplayName "TinyLlama 1.1B Chat" -Repo "litert-community/TinyLlama-1.1B-Chat-v1.0" -Model "TinyLlama-1.1B-Chat-v1.0_multi-prefill-seq_q8_ekv1280.task" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-tinyllama-1.1b-chat-GPU.apk"
    New-LiteRtLmAndroidBenchmark -Name "tinyllama-1.1b-chat-cpu" -DisplayName "TinyLlama 1.1B Chat" -Repo "litert-community/TinyLlama-1.1B-Chat-v1.0" -Model "TinyLlama-1.1B-Chat-v1.0_multi-prefill-seq_q8_ekv1280.task" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-tinyllama-1.1b-chat-CPU.apk"

    New-LiteRtLmAndroidBenchmark -Name "gemma2-2b-it-gpu" -DisplayName "Gemma 2 2B IT" -Repo "litert-community/Gemma2-2B-IT" -Model "Gemma2-2B-IT_multi-prefill-seq_q8_ekv1280.task" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-gemma2-2b-it-GPU.apk"
    New-LiteRtLmAndroidBenchmark -Name "gemma2-2b-it-cpu" -DisplayName "Gemma 2 2B IT" -Repo "litert-community/Gemma2-2B-IT" -Model "Gemma2-2B-IT_multi-prefill-seq_q8_ekv1280.task" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-gemma2-2b-it-CPU.apk"

    New-LiteRtLmAndroidBenchmark -Name "qwen2.5-0.5b-gpu" -DisplayName "Qwen2.5 0.5B Instruct" -Repo "litert-community/Qwen2.5-0.5B-Instruct" -Model "Qwen2.5-0.5B-Instruct-q8.litertlm" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-GPU.apk" -PreferredRank 2 -Notes "User-recommended fast alternative."
    New-LiteRtLmAndroidBenchmark -Name "qwen2.5-0.5b-cpu" -DisplayName "Qwen2.5 0.5B Instruct" -Repo "litert-community/Qwen2.5-0.5B-Instruct" -Model "Qwen2.5-0.5B-Instruct-q8.litertlm" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk" -PreferredRank 2 -Notes "User-recommended fast CPU alternative."

    New-LiteRtLmAndroidBenchmark -Name "qwen2.5-0.5b-task-gpu" -DisplayName "Qwen2.5 0.5B Instruct Task" -Repo "litert-community/Qwen2.5-0.5B-Instruct" -Model "Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task" -Backend "GPU" -Apk "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-task-GPU.apk" -Notes "Task bundle compatibility check."
    New-LiteRtLmAndroidBenchmark -Name "qwen2.5-0.5b-task-cpu" -DisplayName "Qwen2.5 0.5B Instruct Task" -Repo "litert-community/Qwen2.5-0.5B-Instruct" -Model "Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task" -Backend "CPU" -Apk "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-task-CPU.apk" -Notes "Task bundle compatibility check."
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
