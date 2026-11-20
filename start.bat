@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "MV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0runtime\python.exe" set "MV_PYTHON=%~dp0runtime\python.exe"
if not exist "%MV_PYTHON%" (
    echo 未找到 MaskVox 运行环境。源码版请先运行 install.bat。
    pause
    exit /b 1
)
set PYTHONUTF8=1
"%MV_PYTHON%" -m app.main
if errorlevel 1 (
    echo MaskVox 启动失败。请查看上方提示和 logs\maskvox.log。
    pause
    exit /b 1
)
