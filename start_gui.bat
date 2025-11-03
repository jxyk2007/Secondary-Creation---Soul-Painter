@echo off
chcp 65001 >nul
echo ========================================
echo    灵魂画手 AI视频二创工具 v1.0
echo ========================================
echo.
echo 正在启动GUI界面...
echo.

python gui_main_compact.py

if errorlevel 1 (
    echo.
    echo ❌ 启动失败！
    echo.
    echo 可能的原因:
    echo 1. 未安装 Python
    echo 2. 缺少依赖库
    echo.
    echo 请运行: pip install -r requirements.txt
    echo.
    pause
)
