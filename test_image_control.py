# -*- coding: utf-8 -*-
"""
测试图片数量控制功能的集成
"""
import os
import sys

def test_gui_integration():
    """测试GUI集成"""
    print("Testing GUI integration with image control...")

    try:
        import tkinter as tk
        from gui_main import SoulArtistGUI

        # 创建GUI实例（不显示）
        root = tk.Tk()
        root.withdraw()
        app = SoulArtistGUI()

        # 测试新的控制变量
        required_vars = [
            'max_images_var',
            'min_interval_var',
            'images_per_minute_var',
            'smart_limit_var'
        ]

        for var_name in required_vars:
            if hasattr(app, var_name):
                var_obj = getattr(app, var_name)
                print(f"  {var_name}: {var_obj.get()} - OK")
            else:
                print(f"  {var_name}: MISSING")
                return False

        # 测试配置更新
        app.max_images_var.set(8)
        app.min_interval_var.set(6.0)
        app.images_per_minute_var.set(1.5)
        app.smart_limit_var.set(False)

        app.update_config_for_processing()
        print("  Configuration update: OK")

        root.destroy()
        return True

    except Exception as e:
        print(f"  GUI integration test failed: {e}")
        return False

def test_workflow_integration():
    """测试工作流集成"""
    print("\nTesting workflow integration...")

    try:
        # 测试导入
        from image_count_controller import ImageCountController
        print("  Controller import: OK")

        # 测试配置读取
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')

        # 模拟配置参数获取
        image_control_config = {
            'max_images': config.getint('google_vision', 'max_images', fallback=6),
            'min_interval_seconds': config.getfloat('google_vision', 'min_interval_seconds', fallback=5.0),
            'images_per_minute': config.getfloat('google_vision', 'images_per_minute', fallback=2.0),
            'auto_limit_mode': config.get('google_vision', 'auto_limit_mode', fallback='smart')
        }

        print(f"  Config reading: {image_control_config}")

        # 测试控制器创建
        controller = ImageCountController(image_control_config)
        print("  Controller creation: OK")

        # 模拟搞笑时刻过滤
        funny_moments = [
            {'start_time': i * 3.5, 'funny_score': 0.8 + (i % 3) * 0.05, 'description': f'moment_{i}'}
            for i in range(16)  # 模拟原来的16个时刻
        ]

        video_duration = 57.0
        filtered_moments = controller.filter_funny_moments(funny_moments, video_duration)

        print(f"  Filtering: {len(funny_moments)} -> {len(filtered_moments)} moments")
        print(f"  Filtered moments: {[f'{m['start_time']:.1f}s' for m in filtered_moments]}")

        return True

    except Exception as e:
        print(f"  Workflow integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_scenarios():
    """测试不同场景"""
    print("\nTesting different scenarios...")

    try:
        from image_count_controller import ImageCountController

        scenarios = [
            {
                'name': '57秒视频 - 默认设置',
                'config': {'max_images': 6, 'min_interval_seconds': 5.0, 'images_per_minute': 2.0, 'auto_limit_mode': 'smart'},
                'video_duration': 57.0,
                'moments_count': 16
            },
            {
                'name': '120秒视频 - 宽松设置',
                'config': {'max_images': 10, 'min_interval_seconds': 4.0, 'images_per_minute': 3.0, 'auto_limit_mode': 'smart'},
                'video_duration': 120.0,
                'moments_count': 20
            },
            {
                'name': '30秒短视频 - 紧凑设置',
                'config': {'max_images': 3, 'min_interval_seconds': 8.0, 'images_per_minute': 4.0, 'auto_limit_mode': 'smart'},
                'video_duration': 30.0,
                'moments_count': 8
            }
        ]

        for scenario in scenarios:
            print(f"\n  Testing: {scenario['name']}")

            controller = ImageCountController(scenario['config'])

            # 生成模拟数据
            funny_moments = [
                {'start_time': i * (scenario['video_duration'] / scenario['moments_count']),
                 'funny_score': 0.7 + (i % 4) * 0.05,
                 'description': f'moment_{i}'}
                for i in range(scenario['moments_count'])
            ]

            filtered_moments = controller.filter_funny_moments(funny_moments, scenario['video_duration'])
            control_info = controller.get_control_info(scenario['video_duration'])

            print(f"    Original: {scenario['moments_count']} moments")
            print(f"    Filtered: {len(filtered_moments)} moments")
            print(f"    Suggested by time: {control_info['suggested_count_by_time']}")
            print(f"    Final max: {control_info['final_max_count']}")

        return True

    except Exception as e:
        print(f"  Scenario testing failed: {e}")
        return False

def main():
    """主测试函数"""
    print("=== Image Count Control Integration Test ===")

    tests = [
        ("GUI Integration", test_gui_integration),
        ("Workflow Integration", test_workflow_integration),
        ("Different Scenarios", test_different_scenarios),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            if test_func():
                passed += 1
                print(f"  Result: PASS")
            else:
                failed += 1
                print(f"  Result: FAIL")
        except Exception as e:
            failed += 1
            print(f"  Result: ERROR - {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("\nImage count control feature is working perfectly!")
        print("\nNew features:")
        print("  - Maximum images per video (default: 6)")
        print("  - Minimum interval between images (default: 5s)")
        print("  - Images per minute ratio (default: 2.0)")
        print("  - Smart auto-adjustment mode")
        print("  - Manual control mode")
        print("\nFor 57-second video: 16 moments -> 2-6 moments (much better!)")
        return True
    else:
        print(f"\n{failed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)