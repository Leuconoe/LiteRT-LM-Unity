[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity",
    [string[]]$BenchmarkName = @(),
    [string]$SingleApkPath = "",
    [int]$TimeoutSeconds = 600,
    [double]$ThermalMaxCelsius = 45.0,
    [int]$ThermalPollSeconds = 15,
    [switch]$SkipThermalWait,
    [switch]$SkipModelPush,
    [switch]$ClearAppData
)

$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDirectory)
$ApkDirectory = Join-Path $ProjectRoot "Builds\Android"
$LogDirectory = Join-Path $ProjectRoot "Builds\Logs\AndroidDeviceRuns"
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

. (Join-Path $PSScriptRoot "LiteRtLmAndroidBenchmarks.ps1")
$Benchmarks = Select-LiteRtLmAndroidBenchmarks -Name $BenchmarkName

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

function Invoke-AdbBestEffort {
    param(
        [string]$Serial,
        [string[]]$Arguments,
        [string]$WarningMessage = "adb best-effort command failed"
    )

    & adb -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "${WarningMessage}: adb -s $Serial $($Arguments -join ' ') exited with code ${LASTEXITCODE}."
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

function ConvertTo-Megabytes {
    param([object]$Bytes)

    if ($null -eq $Bytes -or [string]::IsNullOrWhiteSpace([string]$Bytes)) {
        return ""
    }

    return [math]::Round(([double]$Bytes / 1MB), 2)
}

function ConvertKilobytesTo-Megabytes {
    param([object]$Kilobytes)

    if ($null -eq $Kilobytes -or [string]::IsNullOrWhiteSpace([string]$Kilobytes)) {
        return ""
    }

    return [math]::Round(([double]$Kilobytes / 1024.0), 2)
}

function Get-ModelSizeBytes {
    param([string]$ModelFileName)

    $modelPath = Join-Path (Join-Path $ProjectRoot "Assets\StreamingAssets") $ModelFileName
    if (!(Test-Path $modelPath)) {
        return ""
    }

    return (Get-Item $modelPath).Length
}

function Push-BenchmarkModel {
    param(
        [string]$Serial,
        [string]$PackageName,
        [string]$ModelFileName
    )

    if ($SkipModelPush) {
        return $false
    }

    $modelPath = Join-Path (Join-Path $ProjectRoot "Assets\StreamingAssets") $ModelFileName
    if (!(Test-Path $modelPath)) {
        throw "Model file not found for device push: $modelPath"
    }

    $deviceDirectory = "/sdcard/Android/data/$PackageName/files/LiteRTLM"
    $devicePath = "$deviceDirectory/$ModelFileName"
    $localSize = (Get-Item $modelPath).Length
    $deviceSize = ((& adb -s $Serial shell stat -c%s $devicePath 2>$null) | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and ![string]::IsNullOrWhiteSpace($deviceSize)) {
        $deviceSize = ([string]$deviceSize).Trim()
        if ($deviceSize -eq [string]$localSize) {
            Write-Host "Model already present on device with matching size: $ModelFileName ($deviceSize bytes)"
            return $false
        }
    }

    Write-Host "Pushing model to device: $ModelFileName -> $devicePath"
    $mkdirOutput = @(Invoke-Adb $Serial @("shell", "mkdir", "-p", $deviceDirectory))
    $mkdirOutput | ForEach-Object { Write-Host $_ }
    $pushOutput = @(Invoke-Adb $Serial @("push", $modelPath, $devicePath))
    $pushOutput | ForEach-Object { Write-Host $_ }
    return $true
}

function Push-BenchmarkConfig {
    param(
        [string]$Serial,
        [string]$PackageName,
        [object]$Benchmark
    )

    $deviceDirectory = "/sdcard/Android/data/$PackageName/files"
    $deviceConfigPath = "$deviceDirectory/LiteRtLmAndroidSmokeTest.config.json"
    $configDirectory = Join-Path $ProjectRoot "temp\android-device-configs"
    New-Item -ItemType Directory -Force -Path $configDirectory | Out-Null

    $configPath = Join-Path $configDirectory "$($Benchmark.Name).json"
    $config = [ordered]@{
        modelPath = $Benchmark.Model
        backend = $Benchmark.Backend
        maxNumTokens = [int]$Benchmark.MaxNumTokens
        maxNumImages = [int]$Benchmark.MaxNumImages
        enableSpeculativeDecoding = [bool]$Benchmark.Speculative
        runStandaloneBenchmark = $true
        benchmarkPrefillTokens = [int]$Benchmark.BenchmarkPrefillTokens
        benchmarkDecodeTokens = 32
        benchmarkRuns = 3
        systemInstruction = "You are a concise Unity Android smoke test assistant."
        resetConversationBeforeEachPrompt = $true
    }

    ($config | ConvertTo-Json -Depth 4) | Set-Content -Path $configPath -Encoding utf8
    Write-Host "Pushing benchmark config: $($Benchmark.Name) -> $deviceConfigPath"
    $mkdirOutput = @(Invoke-Adb $Serial @("shell", "mkdir", "-p", $deviceDirectory))
    $mkdirOutput | ForEach-Object { Write-Host $_ }
    $pushOutput = @(Invoke-Adb $Serial @("push", $configPath, $deviceConfigPath))
    $pushOutput | ForEach-Object { Write-Host $_ }
    return $deviceConfigPath
}

function Get-DeviceStatusPath {
    param([string]$PackageName)

    return "/sdcard/Android/data/$PackageName/files/LiteRtLmAndroidSmokeTest.status.txt"
}

function Get-DeviceStatusText {
    param(
        [string]$Serial,
        [string]$DeviceStatusPath
    )

    try {
        $lines = & adb -s $Serial shell cat $DeviceStatusPath 2>$null
        if ($LASTEXITCODE -ne 0) {
            return ""
        }

        return (@($lines) -join [Environment]::NewLine)
    }
    catch {
        return ""
    }
}

function Test-RunStatusMatched {
    param([string]$StatusText)

    return $StatusText -match "(?m)\]\s+(SUCCESS|FAILURE):"
}

function Get-DeviceThermalSnapshot {
    param([string]$Serial)

    $readings = @()
    $zoneNames = & adb -s $Serial shell ls /sys/class/thermal 2>$null
    foreach ($zoneNameRaw in @($zoneNames)) {
        $zoneName = ([string]$zoneNameRaw).Trim()
        if ($zoneName -notmatch "^thermal_zone[0-9]+$") {
            continue
        }

        $zonePath = "/sys/class/thermal/$zoneName"
        try {
            $name = ((& adb -s $Serial shell cat "$zonePath/type" 2>$null) | Select-Object -First 1)
            $temp = ((& adb -s $Serial shell cat "$zonePath/temp" 2>$null) | Select-Object -First 1)
        }
        catch {
            continue
        }

        if ($null -eq $name -or $null -eq $temp) {
            continue
        }

        $name = ([string]$name).Trim()
        $temp = ([string]$temp).Trim()
        if ([string]::IsNullOrWhiteSpace($name) -or $temp -notmatch "^[0-9]+$") {
            continue
        }

        $raw = [double]$temp
        if ($raw -le 0) {
            continue
        }

        $celsius = if ($raw -gt 1000) { $raw / 1000.0 } else { $raw }
        if ($celsius -lt 1 -or $celsius -gt 120) {
            continue
        }

        $readings += [pscustomobject]@{
            Name = $name
            Celsius = [math]::Round($celsius, 2)
        }
    }

    return $readings
}

function Format-ThermalSnapshot {
    param([object[]]$Readings)

    if ($Readings.Count -eq 0) {
        return ""
    }

    $top = $Readings | Sort-Object Celsius -Descending | Select-Object -First 6
    return (($top | ForEach-Object { "$($_.Name)=$($_.Celsius)C" }) -join "; ")
}

function Get-MaxThermalCelsius {
    param([object[]]$Readings)

    if ($Readings.Count -eq 0) {
        return ""
    }

    return ($Readings | Measure-Object -Property Celsius -Maximum).Maximum
}

function Wait-DeviceThermalReady {
    param([string]$Serial)

    if ($SkipThermalWait) {
        return Get-DeviceThermalSnapshot $Serial
    }

    while ($true) {
        $snapshot = @(Get-DeviceThermalSnapshot $Serial)
        $max = Get-MaxThermalCelsius $snapshot
        $summary = Format-ThermalSnapshot $snapshot

        if ([string]::IsNullOrWhiteSpace([string]$max) -or [double]$max -le $ThermalMaxCelsius) {
            Write-Host "Thermal ready: max=$max C; $summary"
            return $snapshot
        }

        Write-Host "Thermal wait: max=$max C exceeds $ThermalMaxCelsius C. $summary"
        Start-Sleep -Seconds $ThermalPollSeconds
    }
}

function Get-MemInfoSnapshot {
    param(
        [string]$Serial,
        [string]$PackageName,
        [string]$OutputPath
    )

    $meminfo = & adb -s $Serial shell dumpsys meminfo $PackageName 2>$null
    $text = ($meminfo -join [Environment]::NewLine)
    if (![string]::IsNullOrWhiteSpace($OutputPath)) {
        $text | Out-File -FilePath $OutputPath -Encoding utf8
    }

    $totalPss = Get-FirstRegexGroup $text "(?m)^\s*TOTAL\s+([0-9,]+)"
    if ([string]::IsNullOrWhiteSpace($totalPss)) {
        $totalPss = Get-FirstRegexGroup $text "TOTAL PSS:\s*([0-9,]+)"
    }

    $nativeHeap = Get-FirstRegexGroup $text "(?m)^\s*Native Heap\s+([0-9,]+)"
    $javaHeap = Get-FirstRegexGroup $text "(?m)^\s*Java Heap\s+([0-9,]+)"

    [pscustomobject]@{
        TotalPssKb = $totalPss.Replace(",", "")
        NativeHeapPssKb = $nativeHeap.Replace(",", "")
        JavaHeapPssKb = $javaHeap.Replace(",", "")
    }
}

$Serial = Resolve-PhysicalDeviceSerial $DeviceSerial
$DeviceDescription = (& adb -s $Serial shell getprop ro.product.manufacturer).Trim() + " " +
    (& adb -s $Serial shell getprop ro.product.model).Trim() + " Android " +
    (& adb -s $Serial shell getprop ro.build.version.release).Trim()

Write-Host "Device: $Serial ($DeviceDescription)"

$Results = New-Object System.Collections.Generic.List[object]
$csvPath = Join-Path $LogDirectory "$RunId-results.csv"

function Save-BenchmarkResults {
    $script:Results | Export-Csv -NoTypeInformation -Encoding utf8 -Path $script:csvPath
}

foreach ($benchmark in $Benchmarks) {
    $name = $benchmark.Name
    $apkPath = if ([string]::IsNullOrWhiteSpace($SingleApkPath)) {
        Join-Path $ApkDirectory $benchmark.Apk
    } else {
        (Resolve-Path $SingleApkPath).Path
    }
    $modelSizeBytes = Get-ModelSizeBytes $benchmark.Model
    if (-not (Test-Path $apkPath)) {
        throw "APK not found for benchmark '$name': $apkPath"
    }
    $apkName = Split-Path -Leaf $apkPath

    $rawLog = Join-Path $LogDirectory "$RunId-$name.logcat.txt"
    $summaryLog = Join-Path $LogDirectory "$RunId-$name.summary.txt"
    $meminfoLog = Join-Path $LogDirectory "$RunId-$name.meminfo.txt"
    $statusLog = Join-Path $LogDirectory "$RunId-$name.status.txt"
    $deviceStatusPath = Get-DeviceStatusPath $PackageName
    $modelPushed = $false

    Write-Host "=== $name ==="
    $thermalBefore = @(Wait-DeviceThermalReady $Serial)
    $thermalBeforeMax = Get-MaxThermalCelsius $thermalBefore
    $thermalBeforeSummary = Format-ThermalSnapshot $thermalBefore
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
            DisplayName = $benchmark.DisplayName
            Repo = $benchmark.Repo
            Model = $benchmark.Model
            ModelSizeMB = ConvertTo-Megabytes $modelSizeBytes
            Apk = $apkName
            Status = "INSTALL_FAIL"
            Matched = $false
            ModelPushed = $modelPushed
            ModelCopied = ""
            BackendRequested = $benchmark.Backend
            StartedModel = ""
            GpuEvidence = ""
            ThermalBeforeMaxCelsius = $thermalBeforeMax
            ThermalAfterMaxCelsius = ""
            ThermalBeforeSummary = $thermalBeforeSummary
            ThermalAfterSummary = ""
            MemoryTotalPssMB = ""
            MemoryNativeHeapPssMB = ""
            MemoryJavaHeapPssMB = ""
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
            StatusLog = ""
            MemInfoLog = ""
        }) | Out-Null
        Save-BenchmarkResults
        Write-Warning "Install failed for $name with exit code $installExitCode. Continuing with the next benchmark."
        continue
    }

    try {
        if ($ClearAppData) {
            Write-Host "Clearing app data for $PackageName"
            & adb -s $Serial shell pm clear $PackageName | Out-Null
        }

        $modelPushed = Push-BenchmarkModel -Serial $Serial -PackageName $PackageName -ModelFileName $benchmark.Model
        $deviceConfigPath = Push-BenchmarkConfig -Serial $Serial -PackageName $PackageName -Benchmark $benchmark
        Invoke-AdbBestEffort $Serial @("shell", "am", "force-stop", $PackageName) "Pre-run force-stop failed"
        Invoke-AdbBestEffort $Serial @("shell", "rm", "-f", $deviceStatusPath) "Pre-run status cleanup failed"
        Invoke-Adb $Serial @("logcat", "-c")
        Invoke-Adb $Serial @("shell", "monkey", "-p", $PackageName, "-c", "android.intent.category.LAUNCHER", "1")

        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $matched = $false
        $statusText = ""
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 5
            $statusText = Get-DeviceStatusText -Serial $Serial -DeviceStatusPath $deviceStatusPath
            if (![string]::IsNullOrWhiteSpace($statusText)) {
                $statusText | Out-File -FilePath $statusLog -Encoding utf8
                if (Test-RunStatusMatched $statusText) {
                    $matched = $true
                    break
                }
            }

            & adb -s $Serial logcat -d -v threadtime | Out-File -FilePath $rawLog -Encoding utf8
            $content = Get-Content $rawLog -Raw
            if ($content -match "lowmemorykiller: Kill '$([regex]::Escape($PackageName))'" -or
                $content -match "Process $([regex]::Escape($PackageName)).* has died") {
                $matched = $true
                break
            }
        }

        if (![string]::IsNullOrWhiteSpace($statusText)) {
            $statusText | Out-File -FilePath $statusLog -Encoding utf8
        }

        & adb -s $Serial logcat -d -v threadtime | Out-File -FilePath $rawLog -Encoding utf8
        $patterns = "LiteRT-LM AndroidSmoke|MODEL_READY|COPY_MODEL|INITIALIZED|RESPONSE|BENCHMARK|SUCCESS|FAILURE|WebGPU|OpenCL|GPU sampler|GPU|compiled model|Binding size|lowmemorykiller|Process $([regex]::Escape($PackageName)).* has died|AndroidRuntime|FATAL|ERROR"
        Select-String -Path $rawLog -Pattern $patterns | ForEach-Object { $_.Line } | Out-File -FilePath $summaryLog -Encoding utf8
        if (Test-Path $statusLog) {
            Add-Content -Path $summaryLog -Encoding utf8 -Value ""
            Add-Content -Path $summaryLog -Encoding utf8 -Value "STATUS_FILE:"
            Get-Content $statusLog | Add-Content -Path $summaryLog -Encoding utf8
        }

        $summaryText = Get-Content $summaryLog -Raw
        $statusParseText = if (Test-Path $statusLog) { Get-Content $statusLog -Raw } else { $summaryText }
        $status = if ($statusParseText -match "(?m)\]\s+SUCCESS:") { "PASS" } elseif ($statusParseText -match "(?m)\]\s+FAILURE:" -or $summaryText -match "lowmemorykiller|Process $([regex]::Escape($PackageName)).* has died") { "FAIL" } else { "TIMEOUT" }
        $backendRequested = Get-FirstRegexGroup $statusParseText "START: backend=([^,]+)"
        if ([string]::IsNullOrWhiteSpace($backendRequested)) {
            $backendRequested = Get-BackendFromName $name
        }
        $startedModel = Get-FirstRegexGroup $statusParseText "START: backend=[^,]+, model=([^,]+)"
        if ((![string]::IsNullOrWhiteSpace($startedModel) -and ![string]::Equals($startedModel, $benchmark.Model, [StringComparison]::OrdinalIgnoreCase)) -or
            (![string]::IsNullOrWhiteSpace($backendRequested) -and ![string]::Equals($backendRequested, $benchmark.Backend, [StringComparison]::OrdinalIgnoreCase))) {
            $status = "MISMATCH"
        }
        $gpuEvidence = Get-GpuEvidence $summaryText
        $memorySnapshot = Get-MemInfoSnapshot -Serial $Serial -PackageName $PackageName -OutputPath $meminfoLog
        $thermalAfter = @(Get-DeviceThermalSnapshot $Serial)
        $thermalAfterMax = Get-MaxThermalCelsius $thermalAfter
        $thermalAfterSummary = Format-ThermalSnapshot $thermalAfter

        $Results.Add([pscustomobject]@{
            Name = $name
            DisplayName = $benchmark.DisplayName
            Repo = $benchmark.Repo
            Model = $benchmark.Model
            ModelSizeMB = ConvertTo-Megabytes $modelSizeBytes
            Apk = $apkName
            Status = $status
            Matched = $matched
            ModelPushed = $modelPushed
            ModelCopied = Get-FirstRegexGroup $summaryText "MODEL_READY: .*copied=(True|False)"
            BackendRequested = $backendRequested
            StartedModel = $startedModel
            GpuEvidence = $gpuEvidence
            ThermalBeforeMaxCelsius = $thermalBeforeMax
            ThermalAfterMaxCelsius = $thermalAfterMax
            ThermalBeforeSummary = $thermalBeforeSummary
            ThermalAfterSummary = $thermalAfterSummary
            MemoryTotalPssMB = ConvertKilobytesTo-Megabytes $memorySnapshot.TotalPssKb
            MemoryNativeHeapPssMB = ConvertKilobytesTo-Megabytes $memorySnapshot.NativeHeapPssKb
            MemoryJavaHeapPssMB = ConvertKilobytesTo-Megabytes $memorySnapshot.JavaHeapPssKb
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
            StatusLog = $statusLog
            MemInfoLog = $meminfoLog
        }) | Out-Null
        Save-BenchmarkResults

        Get-Content $summaryLog -Tail 80
    }
    catch {
        $Results.Add([pscustomobject]@{
            Name = $name
            DisplayName = $benchmark.DisplayName
            Repo = $benchmark.Repo
            Model = $benchmark.Model
            ModelSizeMB = ConvertTo-Megabytes $modelSizeBytes
            Apk = $apkName
            Status = "RUN_ERROR"
            Matched = $false
            ModelPushed = $modelPushed
            ModelCopied = ""
            BackendRequested = $benchmark.Backend
            StartedModel = ""
            GpuEvidence = ""
            ThermalBeforeMaxCelsius = $thermalBeforeMax
            ThermalAfterMaxCelsius = ""
            ThermalBeforeSummary = $thermalBeforeSummary
            ThermalAfterSummary = ""
            MemoryTotalPssMB = ""
            MemoryNativeHeapPssMB = ""
            MemoryJavaHeapPssMB = ""
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
            StatusLog = $statusLog
            MemInfoLog = $meminfoLog
        }) | Out-Null
        Save-BenchmarkResults
        Write-Warning "Run failed for $name`: $($_.Exception.Message). Continuing with the next benchmark."
    }
}

Save-BenchmarkResults
Write-Host "Results: $csvPath"
$Results | Format-Table -AutoSize
