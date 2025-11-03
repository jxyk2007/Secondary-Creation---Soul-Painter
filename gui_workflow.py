# -*- coding: utf-8 -*-
"""
GUI专用的工作流封装
支持进度回调和状态更新
"""
import os
import logging
from workflow import extract_frames, google_vision_analysis, nano_banana_generate, compose_video_with_overlay
import configparser

class GUIWorkflow:
    """GUI工作流类,支持进度回调"""

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.is_cancelled = False

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 读取配置
        self.config = configparser.ConfigParser()
        self.config.read("config.ini", encoding="utf-8")

    def log_progress(self, percentage, message):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(percentage, message)
        self.logger.info(f"[{percentage}%] {message}")

    def cancel(self):
        """取消处理"""
        self.is_cancelled = True

    def process_video(self, video_path):
        """处理视频的主函数"""
        try:
            self.log_progress(0, "开始处理视频...")

            if self.is_cancelled:
                return None

            # 1. 抽取关键帧
            self.log_progress(10, "正在抽取关键帧...")

            frame_interval = int(self.config.get("video", "frame_interval", fallback="5000"))
            frame_info, video_info = extract_frames(video_path, frame_interval)

            if not frame_info:
                raise Exception("抽帧失败")

            self.log_progress(25, f"成功抽取 {len(frame_info)} 张关键帧")

            if self.is_cancelled:
                return None

            # 2. 分析搞笑程度
            self.log_progress(30, "正在分析视频内容...")

            # 为每个帧添加视频路径信息
            for frame_data in frame_info:
                frame_data["video_path"] = video_path

            funny_score_threshold = float(self.config.get("google_vision", "funny_score_threshold", fallback="0.85"))

            # 使用UniversalVideoAnalyzer进行分析
            funny_frames = self.analyze_video_content(video_path, frame_info, funny_score_threshold)

            if not funny_frames:
                self.log_progress(35, "没有找到搞笑场景,使用所有帧")
                funny_frames = frame_info
            else:
                self.log_progress(35, f"找到 {len(funny_frames)} 个搞笑场景")

            if self.is_cancelled:
                return None

            # 3. 生成手绘风格插画
            self.log_progress(40, "正在生成手绘插画...")

            # 分步显示进度
            total_frames = len(funny_frames)

            def frame_progress_callback(current, total):
                if total > 0:
                    frame_progress = int(40 + (current / total) * 50)
                    self.log_progress(frame_progress, f"生成手绘图 {current}/{total}")

            # 这里需要修改 nano_banana_generate 函数以支持进度回调
            funny_frames_with_illustrations = self.generate_illustrations_with_progress(
                funny_frames, frame_progress_callback
            )

            if self.is_cancelled:
                return None

            self.log_progress(90, "正在合成最终视频...")

            # 4. 合成视频
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"hand_drawn_{video_basename}"

            bgm_volume = float(self.config.get("output", "bgm_volume", fallback="0.5"))

            output_file = compose_video_with_overlay(
                video_info,
                funny_frames_with_illustrations,
                self.config.get("output", "bgm_file", fallback=""),
                self.config.get("output", "draft_dir", fallback="./drafts"),
                output_name,
                overlay_duration=1.0,
                volume=bgm_volume
            )

            if output_file:
                self.log_progress(100, "视频处理完成!")
                return output_file
            else:
                raise Exception("视频合成失败")

        except Exception as e:
            self.log_progress(0, f"处理失败: {e}")
            raise e

    def generate_illustrations_with_progress(self, funny_frames, progress_callback):
        """生成手绘图,支持进度回调"""
        from universal_image_editor import UniversalImageEditor

        # 检测启用的图像生成模式
        enabled_mode = None
        mode_priority = ["qwen", "deepseek", "kimi", "zhipu", "baichuan", "zeroone", "openai", "custom"]

        for mode in mode_priority:
            try:
                if self.config.has_section(mode):
                    enabled_str = self.config.get(mode, "enabled", fallback="false")
                    if enabled_str.lower() in ['true', '1', 'yes', 'on']:
                        enabled_mode = mode
                        break
            except:
                continue

        # 如果没有启用任何模式,检查是否是Gemini模式
        if not enabled_mode:
            try:
                if self.config.has_section("nano_banana"):
                    enable_real = self.config.get("nano_banana", "enable_real_generation", fallback="false")
                    if enable_real.lower() in ['true', '1', 'yes', 'on']:
                        enabled_mode = "gemini"
            except:
                pass

        # 如果还是没有,默认使用模拟模式
        if not enabled_mode:
            enabled_mode = "simulate"
            self.logger.info("未配置任何图像生成模式,使用模拟模式")

        self.logger.info(f"使用图像生成模式: {enabled_mode}")

        # 使用UniversalImageEditor进行生成
        return self._universal_generate_with_progress(funny_frames, progress_callback, enabled_mode)

    def _universal_generate_with_progress(self, funny_frames, progress_callback, mode):
        """使用UniversalImageEditor生成插画,支持进度回调"""
        try:
            from universal_image_editor import UniversalImageEditor
            import cv2

            # 准备API参数
            kwargs = {}

            # 获取API密钥
            if mode == "gemini":
                # Gemini模式使用google_vision配置
                kwargs['api_key'] = self.config.get("google_vision", "api_key", fallback="")
            elif mode in ["qwen", "deepseek", "kimi", "zhipu", "baichuan", "zeroone", "openai", "custom"]:
                # 其他模式使用各自配置
                kwargs['api_key'] = self.config.get(mode, "api_key", fallback="")

                # 如果是自定义模式,需要额外的配置
                if mode == "custom":
                    kwargs['base_url'] = self.config.get(mode, "base_url", fallback="")
                    kwargs['model'] = self.config.get(mode, "model", fallback="dall-e-3")

            # 初始化编辑器
            editor = UniversalImageEditor(mode=mode, **kwargs)

            # 获取提示词配置
            if mode == "qwen":
                hand_drawn_prompt = self.config.get("qwen", "hand_drawn_prompt", fallback="")
            else:
                hand_drawn_prompt = self.config.get("nano_banana", "prompt", fallback="")

            # 测试连接(模拟模式除外)
            if mode != "simulate":
                if not editor.test_connection():
                    self.logger.warning(f"{mode}API连接失败,使用模拟模式")
                    editor = UniversalImageEditor(mode="simulate")
                    mode = "simulate"

            total_frames = len(funny_frames)

            for i, frame_data in enumerate(funny_frames):
                if self.is_cancelled:
                    break

                progress_callback(i, total_frames)

                try:
                    original_path = frame_data["path"]
                    illustration_path = original_path.replace("frames", "illustrations")
                    os.makedirs(os.path.dirname(illustration_path), exist_ok=True)

                    # 获取原图尺寸
                    original_img = cv2.imread(original_path)
                    if original_img is not None:
                        target_height, target_width = original_img.shape[:2]
                        target_size = (target_width, target_height)
                    else:
                        target_size = None

                    # 调用API转换
                    result_base64 = editor.convert_to_hand_drawn(original_path, hand_drawn_prompt)

                    if result_base64:
                        if editor.save_base64_image(result_base64, illustration_path, target_size):
                            self.logger.info(f"✅ {mode}生成手绘图成功: {illustration_path}")
                        else:
                            self._create_fallback_illustration(original_path, illustration_path, frame_data)
                    else:
                        self._create_fallback_illustration(original_path, illustration_path, frame_data)

                    frame_data["illustration_path"] = illustration_path

                except Exception as e:
                    self.logger.error(f"处理帧失败: {e}")
                    self._create_fallback_illustration(original_path, illustration_path, frame_data)
                    frame_data["illustration_path"] = illustration_path

            progress_callback(total_frames, total_frames)
            return funny_frames

        except Exception as e:
            self.logger.error(f"{mode}处理失败: {e},使用模拟模式")
            return self._simulate_generate_with_progress(funny_frames, progress_callback)

    def _simulate_generate_with_progress(self, funny_frames, progress_callback):
        """模拟生成,支持进度回调"""
        total_frames = len(funny_frames)

        for i, frame_data in enumerate(funny_frames):
            if self.is_cancelled:
                break

            progress_callback(i, total_frames)

            original_path = frame_data["path"]
            illustration_path = original_path.replace("frames", "illustrations")
            os.makedirs(os.path.dirname(illustration_path), exist_ok=True)

            self._create_fallback_illustration(original_path, illustration_path, frame_data)
            frame_data["illustration_path"] = illustration_path

        progress_callback(total_frames, total_frames)
        return funny_frames

    def _create_fallback_illustration(self, original_path, illustration_path, frame_data):
        """创建备用手绘图"""
        try:
            import cv2
            import numpy as np

            if os.path.exists(original_path):
                frame = cv2.imread(original_path)
                if frame is not None:
                    # 简单的手绘效果
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.medianBlur(gray, 5)
                    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
                    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    hand_drawn = cv2.bitwise_and(frame, edges_colored)
                    hand_drawn = cv2.convertScaleAbs(hand_drawn, alpha=1.2, beta=20)

                    # 添加标记
                    height, width = hand_drawn.shape[:2]
                    cv2.rectangle(hand_drawn, (10, 10), (300, 80), (255, 255, 255), -1)
                    cv2.rectangle(hand_drawn, (10, 10), (300, 80), (0, 0, 0), 2)
                    cv2.putText(hand_drawn, "Hand-drawn Style", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(hand_drawn, f"t={frame_data.get('timestamp', 0):.1f}s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    cv2.imwrite(illustration_path, hand_drawn)
                    return

        except Exception as e:
            self.logger.error(f"创建备用插画失败: {e}")

        # 创建空白图像作为最后备选
        try:
            import numpy as np
            import cv2
            blank = 250 * np.ones((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Illustration", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 3)
            cv2.imwrite(illustration_path, blank)
        except:
            pass

    def analyze_video_content(self, video_path, frame_info, threshold):
        """使用UniversalVideoAnalyzer分析视频内容"""
        from universal_video_analyzer import UniversalVideoAnalyzer

        # 检测启用的视频分析模式
        enabled_mode = None

        # 优先检查video_analysis section
        if self.config.has_section("video_analysis"):
            mode = self.config.get("video_analysis", "mode", fallback="gemini")
            enabled_str = self.config.get("video_analysis", "enabled", fallback="true")
            if enabled_str.lower() in ['true', '1', 'yes', 'on']:
                enabled_mode = mode

        # 如果没有配置,检查google_vision
        if not enabled_mode:
            if self.config.has_section("google_vision"):
                use_online = self.config.get("google_vision", "use_online_api", fallback="true")
                if use_online.lower() in ['true', '1', 'yes', 'on']:
                    enabled_mode = "gemini"

        # 默认使用模拟模式
        if not enabled_mode:
            enabled_mode = "simulate"
            self.logger.info("未配置视频分析模式,使用模拟模式")

        self.logger.info(f"使用视频分析模式: {enabled_mode}")

        try:
            # 准备API参数
            kwargs = {}

            if enabled_mode == "gemini":
                kwargs['api_key'] = self.config.get("google_vision", "api_key", fallback="")
                kwargs['proxy_enabled'] = self.config.get("google_vision", "proxy_enabled", fallback="false").lower() in ['true', '1']
                kwargs['proxy_host'] = self.config.get("google_vision", "proxy_host", fallback="127.0.0.1")
                kwargs['proxy_port'] = self.config.get("google_vision", "proxy_port", fallback="1080")

            elif enabled_mode == "qwen_video":
                kwargs['api_key'] = self.config.get("video_analysis", "api_key", fallback="")

            elif enabled_mode == "frame_by_frame":
                kwargs['api_key'] = self.config.get("video_analysis", "api_key", fallback="")
                kwargs['base_url'] = self.config.get("video_analysis", "base_url", fallback="")
                kwargs['model'] = self.config.get("video_analysis", "model", fallback="")

            elif enabled_mode == "ollama":
                kwargs['base_url'] = self.config.get("video_analysis", "base_url", fallback="http://localhost:11434")
                kwargs['model'] = self.config.get("video_analysis", "model", fallback="llava")

            # 初始化分析器
            analyzer = UniversalVideoAnalyzer(mode=enabled_mode, **kwargs)

            # 测试连接(模拟模式除外)
            if enabled_mode != "simulate":
                if not analyzer.test_connection():
                    self.logger.warning(f"{enabled_mode}视频分析API连接失败,使用模拟模式")
                    analyzer = UniversalVideoAnalyzer(mode="simulate")

            # 分析视频
            result = analyzer.analyze_video(video_path, frame_info, threshold)

            # 筛选超过阈值的帧
            funny_frames = [f for f in frame_info if f.get("funny_score", 0) >= threshold]

            return funny_frames

        except Exception as e:
            self.logger.error(f"视频分析失败: {e},使用所有帧")
            # 为所有帧设置默认分数
            for frame in frame_info:
                frame["funny_score"] = 0.8
                frame["desc"] = "默认场景"
            return frame_info
