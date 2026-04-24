param(
    [string]$Prompt = "Hello from standalone Windows test",
    [ValidateSet("cpu", "gpu")]
    [string]$Backend = "cpu",
    [string]$ModelPath = "",
    [string]$PromptFilePath = ""
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDirectory)

$ExecutablePath = Join-Path $ScriptDirectory "litert_lm_main.windows_x86_64.exe"
if (-not (Test-Path $ExecutablePath)) {
    throw "LiteRT-LM Windows executable not found: $ExecutablePath"
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $ProjectRoot "Assets\StreamingAssets\model.litertlm"
}

if (-not (Test-Path $ModelPath)) {
    throw "LiteRT-LM model not found: $ModelPath"
}

if (-not [string]::IsNullOrWhiteSpace($PromptFilePath)) {
    if (-not (Test-Path $PromptFilePath)) {
        throw "LiteRT-LM prompt file not found: $PromptFilePath"
    }
}
else {
    $PromptFilePath = Join-Path $env:TEMP "litertlm-standalone-prompt.txt"
    Set-Content -Path $PromptFilePath -Value $Prompt
}

Write-Host "[LiteRT-LM] Executable: $ExecutablePath"
Write-Host "[LiteRT-LM] Model: $ModelPath"
Write-Host "[LiteRT-LM] Backend: $Backend"

& $ExecutablePath --backend=$Backend --model_path=$ModelPath --input_prompt_file=$PromptFilePath
