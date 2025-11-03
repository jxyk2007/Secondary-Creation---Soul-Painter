# -*- coding: utf-8 -*-
"""
灵魂画手GUI启动脚本
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        from gui_main import main
        main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖都已安装")
        input("按Enter键退出...")
    except Exception as e:
        print(f"启动失败: {e}")
        input("按Enter键退出...")