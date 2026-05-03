# Build da imagem e execução com GPU. Vídeo e pasta de saída montados em /data/... dentro do container.
#
# Se o PowerShell bloquear scripts (ExecutionPolicy), na raiz do repo usa:
#   .\scripts\docker-run.cmd -UseOriginal -Video ".\data\input\video.mp4" -Out ".\data\saida" --class-id 2
#
# Na raiz do repositório:
#   .\scripts\docker-run.ps1 -Video ".\data\meu_drone.mp4" -Out ".\data\saida_volta1"
#
# Script na raiz (simples, boa escolha sem ultrapassagem — evita saltos entre karts):
#   .\scripts\docker-run.ps1 -UseOriginal -Video ".\data\so_um_kart.mp4" -Out ".\data\saida" --class-id 2 --conf 0.2
#
# Script inicial (estilo chat + resultado_tracado.png; ver src/extract_line_inicial.py):
#   .\scripts\docker-run.ps1 -UseInicial -Video ".\data\v.mp4" -Out ".\data\out" --class-id -1 --write-overlay
#
# Script em src/ (tracking + lock de ID — melhor com ultrapassagem):
#   .\scripts\docker-run.ps1 -Video ".\v.mp4" -Out ".\out" --conf 0.2 --auto-select --start-x 640 --start-y 400

param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [Parameter(Mandatory = $true)]
    [string]$Out,
    [string]$ImageName = "racing-line-ai",
    [string]$DisplayEnv = $null,
    [switch]$UseOriginal,
    [switch]$UseInicial,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$videoAbs = (Resolve-Path -LiteralPath $Video).Path
$outAbs = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
if (-not (Test-Path -LiteralPath $outAbs)) {
    New-Item -ItemType Directory -Path $outAbs -Force | Out-Null
}
$outAbs = (Resolve-Path -LiteralPath $outAbs).Path

if ($UseOriginal -and $UseInicial) {
    throw "Use apenas um: -UseOriginal ou -UseInicial (ou nenhum para src/extract_line.py)."
}

Write-Host "Build: $ImageName"
docker build -t $ImageName $root

$runArgs = @(
    "run", "--rm", "-it",
    "-v", "${root}:/app",
    "-v", "${videoAbs}:/data/input_video:ro",
    "-v", "${outAbs}:/data/out_dir",
    "--gpus", "all"
)

if ($DisplayEnv) {
    $runArgs += @("-e", "DISPLAY=$DisplayEnv")
}

$scriptPath = if ($UseOriginal) {
    "/app/extract_line.py"
} elseif ($UseInicial) {
    "/app/src/extract_line_inicial.py"
} else {
    "/app/src/extract_line.py"
}
$runArgs += @(
    $ImageName,
    "python", $scriptPath,
    "--video", "/data/input_video",
    "--out", "/data/out_dir"
)

if ($ExtraArgs -and $ExtraArgs.Length -gt 0) {
    $runArgs += $ExtraArgs
}

Write-Host ("docker " + ($runArgs -join " "))
& docker @runArgs
