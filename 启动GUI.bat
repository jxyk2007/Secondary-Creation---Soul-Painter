@echo off
chcp 65001 >nul
title 灵魂画手 - AI视频二创工具

:: 切换到脚本所在目录
cd /d "%~dp0"

echo 🎨 启动灵魂画手GUI界面...
echo 当前目录: %CD%
echo.

:: 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或不在PATH中
    echo 请先安装Python 3.9或更高版本
    pause
    exit /b 1
)

:: 启动GUI
python run_gui.py

echo.
echo 程序已退出
pause