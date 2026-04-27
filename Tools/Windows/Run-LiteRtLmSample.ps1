param(
    [string]$Prompt = "Hello from standalone Windows test",
    [ValidateSet("cpu", "gpu")]
    [string]$Backend = "cpu",
    [string]$ModelPath = "",
    [string]$PromptFilePath = "",
    [string]$SystemMessageFilePath = "",
    [string]$ToolsJsonFilePath = "",
    [string]$MessagesJsonFilePath = "",
    [switch]$EnableConstrainedDecoding,
    [switch]$OutputMessageJson
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

$Arguments = @(
    "--backend=$Backend",
    "--model_path=$ModelPath",
    "--input_prompt_file=$PromptFilePath"
)

if (-not [string]::IsNullOrWhiteSpace($SystemMessageFilePath)) {
    if (-not (Test-Path $SystemMessageFilePath)) {
        throw "LiteRT-LM system message file not found: $SystemMessageFilePath"
    }

    $Arguments += "--system_message_file=$SystemMessageFilePath"
}

if (-not [string]::IsNullOrWhiteSpace($ToolsJsonFilePath)) {
    if (-not (Test-Path $ToolsJsonFilePath)) {
        throw "LiteRT-LM tools JSON file not found: $ToolsJsonFilePath"
    }

    $Arguments += "--tools_json_file=$ToolsJsonFilePath"
}

if (-not [string]::IsNullOrWhiteSpace($MessagesJsonFilePath)) {
    if (-not (Test-Path $MessagesJsonFilePath)) {
        throw "LiteRT-LM messages JSON file not found: $MessagesJsonFilePath"
    }

    $Arguments += "--messages_json_file=$MessagesJsonFilePath"
}

if ($EnableConstrainedDecoding) {
    $Arguments += "--enable_constrained_decoding=true"
}

if ($OutputMessageJson) {
    $Arguments += "--output_message_json=true"
}

& $ExecutablePath @Arguments
