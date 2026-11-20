param([ValidateSet('App', 'Converter')][string]$Mode = 'App')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'

function Run-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "[MaskVox] $([IO.Path]::GetFileName($Exe)) ($LASTEXITCODE)" }
}

function Get-Python310 {
    $versionProbe = 'import struct,sys; sys.exit(sys.version_info[:2] != (3,10) or struct.calcsize("P") != 8)'
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        $exe = & py.exe -3.10-64 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path -LiteralPath $exe)) { return $exe }
    }
    $runtime = Join-Path $root '.runtime'
    $exe = Join-Path $runtime 'tools\python.exe'
    if (!(Test-Path -LiteralPath $exe)) {
        Write-Host '[MaskVox] Downloading Python 3.10 x64 from NuGet...'
        New-Item -ItemType Directory -Force -Path $runtime | Out-Null
        $archive = Join-Path $runtime 'python.3.10.11.zip'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://api.nuget.org/v3-flatcontainer/python/3.10.11/python.3.10.11.nupkg' -OutFile $archive
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash -ne '7C6F99B160A36A7E09492DFCFF2B0A3A60BB5229CA44CDCC3ECB32871A6144D0') {
            throw '[MaskVox] Python SHA256 verification failed.'
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $runtime -Force
    }
    & $exe -c $versionProbe
    if ($LASTEXITCODE -ne 0) { throw '[MaskVox] Python 3.10 x64 is required.' }
    return $exe
}

$venv = if ($Mode -eq 'App') { '.venv' } else { 'converter\.venv' }
$python = Join-Path $root "$venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $python)) {
    Run-Checked (Get-Python310) @('-m', 'venv', (Join-Path $root $venv))
}
if (!$env:PIP_INDEX_URL) { $env:PIP_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple' }
$requirements = if ($Mode -eq 'App') { 'requirements.txt' } else { 'converter\requirements.txt' }

if ($Mode -eq 'Converter') {
    Run-Checked $python @('-m', 'pip', 'install', 'torch==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cpu')
}
Run-Checked $python @('-m', 'pip', 'install', '-r', (Join-Path $root $requirements))
if ($Mode -eq 'Converter') {
    Run-Checked $python @('-m', 'converter.export_base_models')
}
