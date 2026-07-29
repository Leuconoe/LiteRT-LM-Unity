<#
.SYNOPSIS
  Builds KRAFTON's vllm-omni image and serves Raon-Speech-9B-AWQ-INT4 on it.

.DESCRIPTION
  The AWQ-INT4 build cannot load through Windows `transformers`: AWQ is routed to
  `gptqmodel`, which needs `pcre`, and `pypcre` is source-only on Windows with a
  build script that cannot find a Visual Studio generator. Docker sidesteps that
  entirely — the container is Linux, and Docker Desktop's WSL2 backend passes the
  4090 through, which is verified before the build starts.

  This is also the configuration KRAFTON's published RTF numbers come from, so it
  is the only apples-to-apples way to check them.

  The image is FROM vllm/vllm-openai (prebuilt), so nothing compiles from source
  — expect a large pull rather than a long build.

  STATUS: NOT YET RUN END TO END. The GPU probe below is verified working on this
  machine, and the build/serve commands are read off upstream's Dockerfile.ci and
  model card — but the image build was stopped during the base pull, so the
  server has never been started from here and no AWQ number exists yet. Expect to
  debug the serve step on first use rather than trusting it.

.PARAMETER WorkDir
  Where to clone vllm-omni. Defaults under External/, which is untracked scratch
  — this script is the copy that matters, not the clone.

.PARAMETER Serve
  After building, start the server and wait for it to answer.

.PARAMETER Smoke
  Serve, then transcribe one of the project's Korean clips through the OpenAI
  endpoint and print the result. Implies -Serve.

.EXAMPLE
  .\Tools\Research\Raon\Build-RaonVllmOmni.ps1
  .\Tools\Research\Raon\Build-RaonVllmOmni.ps1 -Smoke
#>
[CmdletBinding()]
param(
    [string]$WorkDir = "",
    [string]$Image = "vllm-omni",
    [string]$Model = "KRAFTON/Raon-Speech-9B-AWQ-INT4",
    [int]$Port = 8000,
    [string]$SmokeAudio = "",
    [switch]$SkipBuild,
    [switch]$Serve,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $WorkDir) { $WorkDir = Join-Path $ProjectRoot "External\raon-work" }
if ($Smoke) { $Serve = $true }

Write-Host "== Checking GPU passthrough ==" -ForegroundColor Cyan
$probe = docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 `
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Docker cannot see the GPU. Needs Docker Desktop on the WSL2 backend with " +
          "an NVIDIA driver that supports WSL CUDA. Output: $probe"
}
Write-Host "  $probe" -ForegroundColor Green

if (-not $SkipBuild) {
    New-Item -ItemType Directory -Force $WorkDir | Out-Null
    $repo = Join-Path $WorkDir "vllm-omni"
    if (Test-Path (Join-Path $repo ".git")) {
        Write-Host "== Updating vllm-omni ==" -ForegroundColor Cyan
        git -C $repo pull --ff-only
    }
    else {
        Write-Host "== Cloning vllm-omni ==" -ForegroundColor Cyan
        git clone --depth 1 https://github.com/krafton-ai/vllm-omni.git $repo
    }

    Write-Host "== Building $Image (FROM vllm/vllm-openai — a pull, not a compile) ==" -ForegroundColor Cyan
    docker build -f (Join-Path $repo "docker\Dockerfile.ci") -t $Image $repo
    if ($LASTEXITCODE -ne 0) { throw "docker build failed." }
}

if (-not $Serve) {
    Write-Host "Image ready. Serve with -Serve, or:" -ForegroundColor Green
    Write-Host "  docker run --rm --gpus all --shm-size=16g -p ${Port}:8000 $Image ``"
    Write-Host "    bash -c `"vllm serve $Model --omni --port 8000 --trust-remote-code --quantization awq --dtype float16`""
    exit 0
}

# Reuse the host HF cache so the 7.3 GB download is not repeated per container.
$hfCache = Join-Path $env:USERPROFILE ".cache\huggingface"
New-Item -ItemType Directory -Force $hfCache | Out-Null

$containerName = "raon-vllm-omni"
docker rm -f $containerName 2>$null | Out-Null

Write-Host "== Serving $Model ==" -ForegroundColor Cyan
docker run -d --name $containerName --gpus all --shm-size=16g `
    -p "${Port}:8000" `
    -v "${hfCache}:/root/.cache/huggingface" `
    $Image `
    bash -c "vllm serve $Model --omni --port 8000 --trust-remote-code --quantization awq --dtype float16"
if ($LASTEXITCODE -ne 0) { throw "docker run failed." }

Write-Host "Waiting for the server (model load is minutes, not seconds)…" -ForegroundColor Cyan
$ready = $false
foreach ($attempt in 1..120) {
    Start-Sleep -Seconds 10
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:$Port/v1/models" -TimeoutSec 5
        $ready = $true
        break
    }
    catch {
        # Surface a crashed container rather than waiting out the full 20 minutes.
        $state = docker inspect -f '{{.State.Running}}' $containerName 2>$null
        if ($state -ne "true") {
            docker logs --tail 40 $containerName
            throw "Container exited during startup."
        }
    }
}
if (-not $ready) {
    docker logs --tail 40 $containerName
    throw "Server did not become ready within 20 minutes."
}
Write-Host "Ready: http://localhost:$Port/v1/models" -ForegroundColor Green

if ($Smoke) {
    if (-not $SmokeAudio) {
        $SmokeAudio = Join-Path $ProjectRoot "Assets\StreamingAssets\TestAssets\Audio\현재 서울의 날씨는 흐림 입니다.mp3"
    }
    if (-not (Test-Path $SmokeAudio)) { throw "Smoke clip not found: $SmokeAudio" }

    Write-Host "== STT smoke ==" -ForegroundColor Cyan
    $base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($SmokeAudio))
    $extension = [IO.Path]::GetExtension($SmokeAudio).TrimStart(".")
    $body = @{
        model    = $Model
        messages = @(@{
            role    = "user"
            content = @(
                @{ type = "audio_url"; audio_url = @{ url = "data:audio/$extension;base64,$base64" } },
                @{ type = "text"; text = "Transcribe the audio into text." }
            )
        })
    } | ConvertTo-Json -Depth 8

    $response = Invoke-RestMethod -Uri "http://localhost:$Port/v1/chat/completions" `
        -Method Post -ContentType "application/json; charset=utf-8" `
        -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 300
    Write-Host "  clip: $(Split-Path -Leaf $SmokeAudio)"
    Write-Host "  said: $($response.choices[0].message.content)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Container '$containerName' is still running. Stop it with:" -ForegroundColor Yellow
Write-Host "  docker rm -f $containerName"
