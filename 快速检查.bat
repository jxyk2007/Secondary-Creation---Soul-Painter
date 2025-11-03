@echo off
chcp 65001 >nul
title 灵魂画手环境检查

cd /d "%~dp0"
echo 检查灵魂画手运行环境...
echo.

echo [1/5] 检查Python...
python --version || (echo Python未安装 && pause && exit)

echo [2/5] 检查必要文件...
if not exist "gui_main.py" (echo gui_main.py缺失 && pause && exit)
if not exist "workflow.py" (echo workflow.py缺失 && pause && exit)
if not exist "config.ini" (echo config.ini缺失 && pause && exit)

echo [3/5] 检查FFmpeg...
if exist "ffmpeg.exe" (echo FFmpeg正常) else (echo 警告: ffmpeg.exe未找到)

echo [4/5] 创建工作目录...
if not exist "drafts" mkdir drafts
if not exist "frames" mkdir frames
if not exist "illustrations" mkdir illustrations

echo [5/5] 测试GUI启动...
echo 尝试启动GUI界面...
python run_gui.py

echo.
echo 检查完成
pause