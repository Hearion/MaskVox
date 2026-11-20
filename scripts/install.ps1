param([ValidateSet('App', 'Converter')][string]$Mode = 'App')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'

function Run-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "[MaskVox] 执行 $([IO.Path]::GetFileName($Exe)) 失败，退出码 $LASTEXITCODE。" }
}

function Use-OfficialIndex {
    if (!$defaultIndex -or $env:PIP_INDEX_URL -eq 'https://pypi.org/simple') {
        throw '[MaskVox] 安装依赖失败。请检查网络和上方的 pip 错误。'
    }
    Write-Host '[MaskVox] 正在用官方 PyPI 源重试…'
    $env:PIP_INDEX_URL = 'https://pypi.org/simple'
}

$versionProbe = 'import sys; sys.exit(sys.version_info[:2] != (3,10) or sys.maxsize <= 2**32)'
function Get-Python310 {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        try {
            $exe = & py.exe -3.10-64 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path -LiteralPath $exe)) { return $exe }
        } catch {
            Write-Host '[MaskVox] 正在检查项目内的 Python 运行环境…'
        }
    }
    $runtime = Join-Path $root '.runtime'
    $exe = Join-Path $runtime 'tools\python.exe'
    if (!(Test-Path -LiteralPath $exe)) {
        Write-Host '[MaskVox] 正在从 NuGet 下载 Python 3.10 x64…'
        New-Item -ItemType Directory -Force -Path $runtime | Out-Null
        $archive = Join-Path $runtime 'python.3.10.11.zip'
        if (!(Test-Path -LiteralPath $archive)) {
            Invoke-WebRequest -UseBasicParsing -Uri 'https://api.nuget.org/v3-flatcontainer/python/3.10.11/python.3.10.11.nupkg' -OutFile $archive
        }
        $sha256 = [Security.Cryptography.SHA256]::Create()
        $file = [IO.File]::OpenRead($archive)
        try {
            $hash = [BitConverter]::ToString($sha256.ComputeHash($file)).Replace('-', '')
        } finally {
            $file.Dispose()
            $sha256.Dispose()
        }
        if ($hash -ne '7C6F99B160A36A7E09492DFCFF2B0A3A60BB5229CA44CDCC3ECB32871A6144D0') {
            throw '[MaskVox] Python 安装文件 SHA256 校验失败。'
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $package = [IO.Compression.ZipFile]::OpenRead($archive)
        try {
            foreach ($entry in $package.Entries) {
                if (!$entry.FullName.StartsWith('tools/') -or !$entry.Name) { continue }
                $target = [IO.Path]::GetFullPath((Join-Path $runtime $entry.FullName))
                if (!$target.StartsWith($runtime + '\tools\', [StringComparison]::OrdinalIgnoreCase)) {
                    throw '[MaskVox] Python 安装文件路径异常。'
                }
                [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            }
        } finally {
            $package.Dispose()
        }
    }
    return $exe
}

$venv = if ($Mode -eq 'App') { '.venv' } else { 'converter\.venv' }
$python = Join-Path $root "$venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $python)) {
    Run-Checked (Get-Python310) @('-m', 'venv', (Join-Path $root $venv))
}
& $python -c $versionProbe
if ($LASTEXITCODE -ne 0) { throw '[MaskVox] 请使用 64 位 Python 3.10 创建项目虚拟环境。' }
$defaultIndex = !$env:PIP_INDEX_URL
if ($defaultIndex) { $env:PIP_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple' }
$requirements = if ($Mode -eq 'App') { 'requirements.txt' } else { 'converter\requirements.txt' }

if ($Mode -eq 'Converter') {
    try {
        Run-Checked $python @('-m', 'pip', 'install', '--upgrade', 'pip>=25,<27')
    } catch {
        Use-OfficialIndex
        Run-Checked $python @('-m', 'pip', 'install', '--upgrade', 'pip>=25,<27')
    }
    try {
        Run-Checked $python @('-m', 'pip', 'install', 'torch==2.13.0+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu', '--extra-index-url', $env:PIP_INDEX_URL)
    } catch {
        Use-OfficialIndex
        Run-Checked $python @('-m', 'pip', 'install', 'torch==2.13.0+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu', '--extra-index-url', $env:PIP_INDEX_URL)
    }
}
try {
    Run-Checked $python @('-m', 'pip', 'install', '-r', (Join-Path $root $requirements))
} catch {
    Use-OfficialIndex
    Run-Checked $python @('-m', 'pip', 'install', '-r', (Join-Path $root $requirements))
}
if ($Mode -eq 'Converter') {
    Run-Checked $python @('-m', 'converter.export_base_models')
}
