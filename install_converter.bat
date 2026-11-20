@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -Mode Converter
if errorlevel 1 (
    echo 转换组件安装失败。请查看 README 中的安装说明。
    pause
    exit /b 1
)
echo 转换组件和基础模型安装完成。
if /i not "%~1"=="--no-pause" pause
