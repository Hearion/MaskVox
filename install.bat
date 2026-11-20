@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -Mode App
if errorlevel 1 (
    echo MaskVox 安装失败。请检查上方提示。
    pause
    exit /b 1
)
echo 主程序安装完成。双击 start.bat 启动。
if /i not "%~1"=="--no-pause" pause
