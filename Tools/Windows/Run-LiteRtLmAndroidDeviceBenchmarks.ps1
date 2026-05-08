param(
    [string]$DeviceSerial = "",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity",
    [string[]]$BenchmarkName = @(),
    [int]$TimeoutSeconds = 600,
    [switch]$ClearAppData
)

$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDirectory)
$ApkDirectory = Join-Path $ProjectRoot "Builds\Android"
$LogDirectory = Join-Path $ProjectRoot "Builds\Logs\AndroidDeviceRuns"
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

$Benchmarks = @(
    @{ Name = "gemma-4-E2B-it-gpu"; Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it.apk" },
    @{ Name = "gemma-4-E2B-it-gpu-nospec"; Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-nospec.apk" },
    @{ Name = "gemma-4-E2B-it-cpu"; Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-CPU.apk" },
    @{ Name = "gemma3-1b-it-gpu"; Apk = "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4.apk" },
    @{ Name = "gemma3-1b-it-cpu"; Apk = "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-CPU.apk" },
    @{ Name = "gemma3-270m-it-gpu"; Apk = "LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8.apk" },
    @{ Name = "mobile-actions-gpu"; Apk = "LiteRtLmAndroidSmokeTest-mobile_actions_q8_ekv1024.apk" },
    @{ Name = "qwen3-0.6b-gpu"; Apk = "LiteRtLmAndroidSmokeTest-Qwen3-0.6B.apk" },
    @{ Name = "qwen2.5-0.5b-gpu"; Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct.apk" },
    @{ Name = "qwen2.5-0.5b-cpu"; Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk" },
    @{ Name = "qwen2.5-1.5b-gpu"; Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct.apk" }
)

if ($BenchmarkName.Count -gt 0) {
    $selected = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $BenchmarkName) {
        [void]$selected.Add($name)
    }

    $Benchmarks = @($Benchmarks | Where-Object { $selected.Contains($_.Name) })
    if ($Benchmarks.Count -eq 0) {
        throw "No benchmarks matched -BenchmarkName: $($BenchmarkName -join ', ')"
    }
}

function Get-AdbDeviceLines {
    $lines = & adb devices -l
    return @($lines | Select-Object -Skip 1 | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Resolve-PhysicalDeviceSerial {
    param([string]$ConfiguredSerial)

    $deviceLines = Get-AdbDeviceLines
    if (-not [string]::IsNullOrWhiteSpace($ConfiguredSerial)) {
        $line = $deviceLines | Where-Object { $_ -match "^$([regex]::Escape($ConfiguredSerial))\s+device\s" } | Select-Object -First 1
        if ($null -eq $line) {
            throw "Configured Android device is not connected or not authorized: $ConfiguredSerial"
        }

        if ($ConfiguredSerial -like "emulator-*") {
            throw "Refusing to run physical-device benchmarks on emulator serial: $ConfiguredSerial"
        }

        return $ConfiguredSerial
    }

    $physicalDevices = @($deviceLines | Where-Object { $_ -notmatch "^emulator-" -and $_ -match "\sdevice\s" })
    if ($physicalDevices.Count -eq 0) {
        throw "No physical Android device is connected. Current adb devices:`n$($deviceLines -join [Environment]::NewLine)"
    }

    if ($physicalDevices.Count -gt 1) {
        throw "Multiple physical Android devices found. Pass -DeviceSerial. Devices:`n$($physicalDevices -join [Environment]::NewLine)"
    }

    return ($physicalDevices[0] -split "\s+")[0]
}

function Invoke-Adb {
    param(
        [string]$Serial,
        [string[]]$Arguments
    )

    & adb -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: adb -s $Serial $($Arguments -join ' ')"
    }
}

function Get-FirstRegexGroup {
    param(
        [string]$Text,
        [string]$Pattern
    )

    $match = [regex]::Match($Text, $Pattern)
    if ($match.Success) {
        return $match.Groups[1].Value
    }

    return ""
}

function Get-BackendFromName {
    param([string]$Name)

    if ($Name -match "-cpu$") {
        return "CPU"
    }

    if ($Name -match "-gpu$") {
        return "GPU"
    }

    return ""
}

function Get-GpuEvidence {
    param([string]$SummaryText)

    $evidence = New-Object System.Collections.Generic.List[string]
    if ($SummaryText -match "Initializing OpenCL-based API|delegate_opencl") {
        $evidence.Add("NativeOpenCL")
    }
    if ($SummaryText -match "Created a WebGPU environment|delegate_webgpu|Using WebGPU instead") {
        $evidence.Add("WebGPU")
    }
    if ($SummaryText -match "Dynamically loaded LiteRtTopKOpenClSampler|Statically linked LiteRtTopKOpenClSampler") {
        $evidence.Add("OpenCLSampler")
    }
    if ($SummaryText -match "Dynamically loaded LiteRtTopKWebGpuSampler|Statically linked LiteRtTopKWebGpuSampler") {
        $evidence.Add("WebGPUSampler")
    }
    if ($SummaryText -match "GPU sampler unavailable|Falling back to CPU sampling") {
        $evidence.Add("CpuSamplerFallback")
    }
    if ($SummaryText -match "backend=GPU" -and $evidence.Count -eq 0) {
        $evidence.Add("RequestedGPU")
    }

    return ($evidence -join "+")
}

$Serial = Resolve-PhysicalDeviceSerial $DeviceSerial
$DeviceDescription = (& adb -s $Serial shell getprop ro.product.manufacturer).Trim() + " " +
    (& adb -s $Serial shell getprop ro.product.model).Trim() + " Android " +
    (& adb -s $Serial shell getprop ro.build.version.release).Trim()

Write-Host "Device: $Serial ($DeviceDescription)"

$Results = New-Object System.Collections.Generic.List[object]

foreach ($benchmark in $Benchmarks) {
    $name = $benchmark.Name
    $apkPath = Join-Path $ApkDirectory $benchmark.Apk
    if (-not (Test-Path $apkPath)) {
        throw "APK not found for benchmark '$name': $apkPath"
    }

    $rawLog = Join-Path $LogDirectory "$RunId-$name.logcat.txt"
    $summaryLog = Join-Path $LogDirectory "$RunId-$name.summary.txt"

    Write-Host "=== $name ==="
    Write-Host "Installing $apkPath"
    $installOutput = @()
    $installExitCode = 1
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        & adb start-server | Out-Null
        $installOutput = & adb -s $Serial install -r -d -t $apkPath 2>&1
        $installExitCode = $LASTEXITCODE
        if ($installExitCode -eq 0) {
            break
        }

        Write-Warning "Install attempt $attempt failed for $name with exit code $installExitCode."
        Start-Sleep -Seconds 3
    }

    $installOutput | ForEach-Object { Write-Host $_ }
    if ($installExitCode -ne 0) {
        $Results.Add([pscustomobject]@{
            Name = $name
            Apk = $benchmark.Apk
            Status = "INSTALL_FAIL"
            Matched = $false
            ModelCopied = ""
            BackendRequested = Get-BackendFromName $name
            GpuEvidence = ""
            InitSeconds = ""
            Turn1Seconds = ""
            Turn2Seconds = ""
            BenchmarkAverageSeconds = ""
            BenchmarkInitMs = ""
            BenchmarkTimeToFirstTokenSeconds = ""
            BenchmarkPrefillTokensPerSecond = ""
            BenchmarkDecodeTokensPerSecond = ""
            FunctionCallingHitRate = ""
            TotalSeconds = ""
            RawLog = ""
            SummaryLog = ""
        }) | Out-Null
        Write-Warning "Install failed for $name with exit code $installExitCode. Continuing with the next benchmark."
        continue
    }

    try {
        if ($ClearAppData) {
            Write-Host "Clearing app data for $PackageName"
            & adb -s $Serial shell pm clear $PackageName | Out-Null
        }

        Invoke-Adb $Serial @("shell", "am", "force-stop", $PackageName)
        Invoke-Adb $Serial @("logcat", "-c")
        Invoke-Adb $Serial @("shell", "monkey", "-p", $PackageName, "-c", "android.intent.category.LAUNCHER", "1")

        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $matched = $false
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 5
            & adb -s $Serial logcat -d -v threadtime | Out-File -FilePath $rawLog -Encoding utf8
            $content = Get-Content $rawLog -Raw
        if ($content -match "\[LiteRT-LM AndroidSmoke\] (SUCCESS|FAILURE)" -or
            $content -match "lowmemorykiller: Kill '$([regex]::Escape($PackageName))'" -or
            $content -match "Process $([regex]::Escape($PackageName)).* has died") {
                $matched = $true
                break
            }
        }

        $patterns = "LiteRT-LM AndroidSmoke|MODEL_READY|COPY_MODEL|INITIALIZED|RESPONSE|BENCHMARK|SUCCESS|FAILURE|WebGPU|OpenCL|GPU sampler|GPU|compiled model|Binding size|lowmemorykiller|Process $([regex]::Escape($PackageName)).* has died|AndroidRuntime|FATAL|ERROR"
        Select-String -Path $rawLog -Pattern $patterns | ForEach-Object { $_.Line } | Out-File -FilePath $summaryLog -Encoding utf8
        $summaryText = Get-Content $summaryLog -Raw
        $status = if ($summaryText -match "\[LiteRT-LM AndroidSmoke\] SUCCESS") { "PASS" } elseif ($summaryText -match "\[LiteRT-LM AndroidSmoke\] FAILURE" -or $summaryText -match "lowmemorykiller|Process $([regex]::Escape($PackageName)).* has died") { "FAIL" } else { "TIMEOUT" }
        $backendRequested = Get-FirstRegexGroup $summaryText "START: backend=([^,]+)"
        if ([string]::IsNullOrWhiteSpace($backendRequested)) {
            $backendRequested = Get-BackendFromName $name
        }
        $gpuEvidence = Get-GpuEvidence $summaryText

        $Results.Add([pscustomobject]@{
            Name = $name
            Apk = $benchmark.Apk
            Status = $status
            Matched = $matched
            ModelCopied = Get-FirstRegexGroup $summaryText "MODEL_READY: .*copied=(True|False)"
            BackendRequested = $backendRequested
            GpuEvidence = $gpuEvidence
            InitSeconds = Get-FirstRegexGroup $summaryText "INITIALIZED: .*elapsedSeconds=([0-9.]+)"
            Turn1Seconds = Get-FirstRegexGroup $summaryText "RESPONSE: 1/2: elapsedSeconds=([0-9.]+)"
            Turn2Seconds = Get-FirstRegexGroup $summaryText "RESPONSE: 2/2: elapsedSeconds=([0-9.]+)"
            BenchmarkAverageSeconds = Get-FirstRegexGroup $summaryText "BENCHMARK_SUMMARY: .*averageElapsedSeconds=([0-9.]+)"
            BenchmarkInitMs = Get-FirstRegexGroup $summaryText "BENCHMARK_RESULT: .*Init Total: ([0-9.]+) ms"
            BenchmarkTimeToFirstTokenSeconds = Get-FirstRegexGroup $summaryText "BENCHMARK_RESULT: .*Time to first token: ([0-9.]+) s"
            BenchmarkPrefillTokensPerSecond = Get-FirstRegexGroup $summaryText "BENCHMARK_RESULT: .*Last Prefill: [0-9]+ tokens, ([0-9.]+) tokens/sec"
            BenchmarkDecodeTokensPerSecond = Get-FirstRegexGroup $summaryText "BENCHMARK_RESULT: .*Last Decode: [0-9]+ tokens, ([0-9.]+) tokens/sec"
            FunctionCallingHitRate = ""
            TotalSeconds = Get-FirstRegexGroup $summaryText "SUCCESS: .*totalElapsedSeconds=([0-9.]+)"
            RawLog = $rawLog
            SummaryLog = $summaryLog
        }) | Out-Null

        Get-Content $summaryLog -Tail 80
    }
    catch {
        $Results.Add([pscustomobject]@{
            Name = $name
            Apk = $benchmark.Apk
            Status = "RUN_ERROR"
            Matched = $false
            ModelCopied = ""
            BackendRequested = Get-BackendFromName $name
            GpuEvidence = ""
            InitSeconds = ""
            Turn1Seconds = ""
            Turn2Seconds = ""
            BenchmarkAverageSeconds = ""
            BenchmarkInitMs = ""
            BenchmarkTimeToFirstTokenSeconds = ""
            BenchmarkPrefillTokensPerSecond = ""
            BenchmarkDecodeTokensPerSecond = ""
            FunctionCallingHitRate = ""
            TotalSeconds = ""
            RawLog = $rawLog
            SummaryLog = $summaryLog
        }) | Out-Null
        Write-Warning "Run failed for $name`: $($_.Exception.Message). Continuing with the next benchmark."
    }
}

$csvPath = Join-Path $LogDirectory "$RunId-results.csv"
$Results | Export-Csv -NoTypeInformation -Encoding utf8 -Path $csvPath
Write-Host "Results: $csvPath"
$Results | Format-Table -AutoSize
