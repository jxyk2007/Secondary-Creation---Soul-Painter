# -*- coding: utf-8 -*-
"""
灵魂画手 GUI 主界面
基于tkinter的图形用户界面，适合小白用户使用
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import configparser
from datetime import datetime
import subprocess

# 导入核心功能
from gui_workflow import GUIWorkflow
from qwen_image_editor import QwenImageEditor

class SoulArtistGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("灵魂画手 v1.0 - AI视频二创工具")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # 设置图标
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass  # 图标文件不存在时忽略

        # 配置管理
        self.config = configparser.ConfigParser()
        self.config_file = "config.ini"
        self.load_config()

        # 状态变量
        self.video_file = tk.StringVar()
        self.video_dir = tk.StringVar()
        self.processing_mode = tk.StringVar(value="single")  # single 或 batch
        self.mode_var = tk.StringVar(value="qwen")
        self.api_key_var = tk.StringVar()
        self.frame_interval_var = tk.IntVar(value=5000)
        self.threshold_var = tk.DoubleVar(value=0.85)
        self.volume_var = tk.DoubleVar(value=0.5)

        # 进度相关
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="准备就绪")
        self.is_processing = False

        # 消息队列（用于线程间通信）
        self.message_queue = queue.Queue()

        # 创建界面
        self.create_widgets()
        self.setup_layout()
        self.load_settings()

        # 启动消息处理
        self.process_messages()

    def load_config(self):
        """加载配置文件"""
        try:
            self.config.read(self.config_file, encoding="utf-8")
        except Exception as e:
            print(f"加载配置文件失败: {e}")

    def save_config(self):
        """保存配置文件"""
        try:
            # 更新配置
            if not self.config.has_section("qwen"):
                self.config.add_section("qwen")

            self.config.set("qwen", "api_key", self.api_key_var.get())
            self.config.set("qwen", "enabled", "true" if self.mode_var.get() == "qwen" else "false")

            if not self.config.has_section("video"):
                self.config.add_section("video")
            self.config.set("video", "frame_interval", str(self.frame_interval_var.get()))

            if not self.config.has_section("google_vision"):
                self.config.add_section("google_vision")
            self.config.set("google_vision", "funny_score_threshold", str(self.threshold_var.get()))

            if not self.config.has_section("output"):
                self.config.add_section("output")
            self.config.set("output", "bgm_volume", str(self.volume_var.get()))

            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)

        except Exception as e:
            self.log_message(f"保存配置失败: {e}")

    def load_settings(self):
        """从配置文件加载设置"""
        try:
            if self.config.has_option("qwen", "api_key"):
                self.api_key_var.set(self.config.get("qwen", "api_key"))

            if self.config.has_option("video", "frame_interval"):
                self.frame_interval_var.set(int(self.config.get("video", "frame_interval")))

            if self.config.has_option("google_vision", "funny_score_threshold"):
                self.threshold_var.set(float(self.config.get("google_vision", "funny_score_threshold")))

            if self.config.has_option("output", "bgm_volume"):
                self.volume_var.set(float(self.config.get("output", "bgm_volume")))

        except Exception as e:
            self.log_message(f"加载设置失败: {e}")

    def create_widgets(self):
        """创建界面组件"""

        # 主标题
        title_frame = ttk.Frame(self.root)
        title_label = ttk.Label(title_frame, text="🎨 灵魂画手 v1.0", font=("Microsoft YaHei", 16, "bold"))
        title_label.pack(pady=10)

        # 文件选择区域
        file_frame = ttk.LabelFrame(self.root, text="📁 视频文件选择", padding=10)

        # 处理模式选择
        mode_select_frame = ttk.Frame(file_frame)
        self.single_radio = ttk.Radiobutton(mode_select_frame, text="单个文件",
                                           variable=self.processing_mode, value="single",
                                           command=self.on_processing_mode_change)
        self.single_radio.pack(side=tk.LEFT, padx=(0, 20))

        self.batch_radio = ttk.Radiobutton(mode_select_frame, text="批量处理（选择文件夹）",
                                          variable=self.processing_mode, value="batch",
                                          command=self.on_processing_mode_change)
        self.batch_radio.pack(side=tk.LEFT)
        mode_select_frame.pack(fill=tk.X, pady=(0, 10))

        # 文件/目录选择
        file_select_frame = ttk.Frame(file_frame)
        self.file_entry = ttk.Entry(file_select_frame, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

        self.browse_btn = ttk.Button(file_select_frame, text="浏览文件", command=self.browse_file)
        self.browse_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self.browse_dir_btn = ttk.Button(file_select_frame, text="选择文件夹", command=self.browse_directory)
        self.browse_dir_btn.pack(side=tk.RIGHT)

        # 文件列表显示（批量模式时显示）
        self.file_list_frame = ttk.Frame(file_frame)
        self.file_list_label = ttk.Label(self.file_list_frame, text="找到的视频文件:")
        self.file_list_text = tk.Text(self.file_list_frame, height=4, width=60)
        self.file_list_scrollbar = ttk.Scrollbar(self.file_list_frame, orient=tk.VERTICAL, command=self.file_list_text.yview)
        self.file_list_text.configure(yscrollcommand=self.file_list_scrollbar.set)

        # 模式选择区域
        mode_frame = ttk.LabelFrame(self.root, text="🤖 AI模式选择", padding=10)

        # 传统模式
        traditional_frame = ttk.LabelFrame(mode_frame, text="传统模式", padding=5)
        traditional_frame.pack(fill=tk.X, pady=(0, 10))

        self.qwen_radio = ttk.Radiobutton(traditional_frame, text="通义千问 (推荐)",
                                         variable=self.mode_var, value="qwen",
                                         command=self.on_mode_change)
        self.qwen_radio.pack(anchor=tk.W, pady=2)

        self.gemini_radio = ttk.Radiobutton(traditional_frame, text="Gemini (需代理)",
                                           variable=self.mode_var, value="gemini",
                                           command=self.on_mode_change)
        self.gemini_radio.pack(anchor=tk.W, pady=2)

        self.simulate_radio = ttk.Radiobutton(traditional_frame, text="模拟模式 (无需网络)",
                                             variable=self.mode_var, value="simulate",
                                             command=self.on_mode_change)
        self.simulate_radio.pack(anchor=tk.W, pady=2)

        # OpenAI兼容模式
        openai_frame = ttk.LabelFrame(mode_frame, text="OpenAI兼容模式", padding=5)
        openai_frame.pack(fill=tk.X, pady=(0, 5))

        # 第一行
        row1_frame = ttk.Frame(openai_frame)
        row1_frame.pack(fill=tk.X, pady=2)

        self.deepseek_radio = ttk.Radiobutton(row1_frame, text="DeepSeek",
                                             variable=self.mode_var, value="deepseek",
                                             command=self.on_mode_change)
        self.deepseek_radio.pack(side=tk.LEFT, padx=(0, 20))

        self.kimi_radio = ttk.Radiobutton(row1_frame, text="月之暗面(Kimi)",
                                         variable=self.mode_var, value="kimi",
                                         command=self.on_mode_change)
        self.kimi_radio.pack(side=tk.LEFT, padx=(0, 20))

        self.zhipu_radio = ttk.Radiobutton(row1_frame, text="智谱AI",
                                          variable=self.mode_var, value="zhipu",
                                          command=self.on_mode_change)
        self.zhipu_radio.pack(side=tk.LEFT)

        # 第二行
        row2_frame = ttk.Frame(openai_frame)
        row2_frame.pack(fill=tk.X, pady=2)

        self.baichuan_radio = ttk.Radiobutton(row2_frame, text="百川智能",
                                             variable=self.mode_var, value="baichuan",
                                             command=self.on_mode_change)
        self.baichuan_radio.pack(side=tk.LEFT, padx=(0, 20))

        self.zeroone_radio = ttk.Radiobutton(row2_frame, text="01.AI(零一万物)",
                                            variable=self.mode_var, value="zeroone",
                                            command=self.on_mode_change)
        self.zeroone_radio.pack(side=tk.LEFT, padx=(0, 20))

        self.openai_radio = ttk.Radiobutton(row2_frame, text="OpenAI官方",
                                           variable=self.mode_var, value="openai",
                                           command=self.on_mode_change)
        self.openai_radio.pack(side=tk.LEFT)

        # 第三行
        row3_frame = ttk.Frame(openai_frame)
        row3_frame.pack(fill=tk.X, pady=2)

        self.custom_radio = ttk.Radiobutton(row3_frame, text="自定义OpenAI兼容",
                                           variable=self.mode_var, value="custom",
                                           command=self.on_mode_change)
        self.custom_radio.pack(side=tk.LEFT)

        # API配置区域
        api_frame = ttk.LabelFrame(self.root, text="🔑 API配置", padding=10)

        # API密钥配置
        api_key_frame = ttk.Frame(api_frame)
        api_key_frame.pack(fill=tk.X, pady=(0, 5))

        self.api_label = ttk.Label(api_key_frame, text="API密钥:")
        self.api_label.pack(side=tk.LEFT, pady=2)

        self.api_entry = ttk.Entry(api_key_frame, textvariable=self.api_key_var, width=50)
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # 自定义配置区域（默认隐藏）
        self.custom_config_frame = ttk.LabelFrame(api_frame, text="自定义配置", padding=5)

        # Base URL配置
        base_url_frame = ttk.Frame(self.custom_config_frame)
        base_url_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(base_url_frame, text="Base URL:").pack(side=tk.LEFT)
        self.base_url_var = tk.StringVar()
        self.base_url_entry = ttk.Entry(base_url_frame, textvariable=self.base_url_var, width=40)
        self.base_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # 模型配置
        model_frame = ttk.Frame(self.custom_config_frame)
        model_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(model_frame, text="模型名称:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_var, width=30)
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # 测试API按钮
        test_frame = ttk.Frame(api_frame)
        test_frame.pack(fill=tk.X, pady=(5, 0))

        self.test_api_btn = ttk.Button(test_frame, text="测试API连接", command=self.test_api)
        self.test_api_btn.pack(side=tk.LEFT)

        # 连接状态显示
        self.connection_status_var = tk.StringVar(value="未测试")
        self.connection_status_label = ttk.Label(test_frame, textvariable=self.connection_status_var, foreground="gray")
        self.connection_status_label.pack(side=tk.LEFT, padx=(10, 0))

        # 高级设置区域
        settings_frame = ttk.LabelFrame(self.root, text="⚙️ 高级设置", padding=10)

        # 抽帧间隔
        interval_frame = ttk.Frame(settings_frame)
        ttk.Label(interval_frame, text="抽帧间隔:").pack(side=tk.LEFT)
        interval_spinbox = ttk.Spinbox(interval_frame, from_=1000, to=10000, width=10,
                                      textvariable=self.frame_interval_var)
        interval_spinbox.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Label(interval_frame, text="毫秒").pack(side=tk.LEFT)
        interval_frame.pack(fill=tk.X, pady=2)

        # 搞笑阈值
        threshold_frame = ttk.Frame(settings_frame)
        ttk.Label(threshold_frame, text="搞笑阈值:").pack(side=tk.LEFT)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0,
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.threshold_label = ttk.Label(threshold_frame, text="0.85")
        self.threshold_label.pack(side=tk.LEFT)
        threshold_scale.configure(command=self.update_threshold_label)
        threshold_frame.pack(fill=tk.X, pady=2)

        # BGM音量
        volume_frame = ttk.Frame(settings_frame)
        ttk.Label(volume_frame, text="BGM音量:").pack(side=tk.LEFT)
        volume_scale = ttk.Scale(volume_frame, from_=0.0, to=1.0,
                                variable=self.volume_var, orient=tk.HORIZONTAL)
        volume_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.volume_label = ttk.Label(volume_frame, text="0.5")
        self.volume_label.pack(side=tk.LEFT)
        volume_scale.configure(command=self.update_volume_label)
        volume_frame.pack(fill=tk.X, pady=2)

        # 图片数量控制
        image_control_frame = ttk.LabelFrame(settings_frame, text="图片生成控制", padding=5)
        image_control_frame.pack(fill=tk.X, pady=(10, 0))

        # 最大图片数量
        max_images_frame = ttk.Frame(image_control_frame)
        ttk.Label(max_images_frame, text="最大图片数:").pack(side=tk.LEFT)
        self.max_images_var = tk.IntVar(value=6)
        max_images_spinbox = ttk.Spinbox(max_images_frame, from_=1, to=20, width=10,
                                        textvariable=self.max_images_var)
        max_images_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(max_images_frame, text="张").pack(side=tk.LEFT)
        max_images_frame.pack(fill=tk.X, pady=2)

        # 最小间隔时间
        min_interval_frame = ttk.Frame(image_control_frame)
        ttk.Label(min_interval_frame, text="最小间隔:").pack(side=tk.LEFT)
        self.min_interval_var = tk.DoubleVar(value=5.0)
        min_interval_spinbox = ttk.Spinbox(min_interval_frame, from_=1.0, to=30.0,
                                          increment=0.5, width=10,
                                          textvariable=self.min_interval_var)
        min_interval_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(min_interval_frame, text="秒").pack(side=tk.LEFT)
        min_interval_frame.pack(fill=tk.X, pady=2)

        # 每分钟图片数
        images_per_minute_frame = ttk.Frame(image_control_frame)
        ttk.Label(images_per_minute_frame, text="每分钟:").pack(side=tk.LEFT)
        self.images_per_minute_var = tk.DoubleVar(value=2.0)
        images_per_minute_spinbox = ttk.Spinbox(images_per_minute_frame, from_=0.5, to=10.0,
                                               increment=0.5, width=10,
                                               textvariable=self.images_per_minute_var)
        images_per_minute_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(images_per_minute_frame, text="张").pack(side=tk.LEFT)
        images_per_minute_frame.pack(fill=tk.X, pady=2)

        # 智能限制模式
        smart_mode_frame = ttk.Frame(image_control_frame)
        self.smart_limit_var = tk.BooleanVar(value=True)
        smart_checkbox = ttk.Checkbutton(smart_mode_frame, text="智能限制模式 (自动调整密度)",
                                        variable=self.smart_limit_var)
        smart_checkbox.pack(side=tk.LEFT)
        smart_mode_frame.pack(fill=tk.X, pady=2)

        # 操作按钮区域
        action_frame = ttk.Frame(self.root)

        self.start_btn = ttk.Button(action_frame, text="🎬 开始生成",
                                   command=self.start_processing, style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.open_output_btn = ttk.Button(action_frame, text="📁 打开输出目录",
                                         command=self.open_output_dir)
        self.open_output_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(action_frame, text="⏹️ 停止",
                                  command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))

        # 进度显示区域
        progress_frame = ttk.LabelFrame(self.root, text="📊 处理进度", padding=10)

        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W)

        # 批量处理统计信息框架
        self.batch_stats_frame = ttk.Frame(progress_frame)
        self.batch_stats_text = tk.Text(self.batch_stats_frame, height=3, width=50,
                                       font=("微软雅黑", 9), state=tk.DISABLED)
        self.batch_stats_text.pack(fill=tk.X)
        # 默认隐藏批量统计信息
        self.batch_stats_frame.pack_forget()

        # 日志输出区域
        log_frame = ttk.LabelFrame(self.root, text="📝 日志输出", padding=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=70)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 存储组件引用
        self.title_frame = title_frame
        self.file_frame = file_frame
        self.mode_select_frame = mode_select_frame
        self.file_select_frame = file_select_frame
        self.mode_frame = mode_frame
        self.api_frame = api_frame
        self.settings_frame = settings_frame
        self.action_frame = action_frame
        self.progress_frame = progress_frame
        self.log_frame = log_frame

    def setup_layout(self):
        """设置布局"""
        self.title_frame.pack(fill=tk.X, padx=10, pady=5)
        self.file_frame.pack(fill=tk.X, padx=10, pady=5)
        self.file_select_frame.pack(fill=tk.X)
        # 文件列表默认隐藏
        self.file_list_frame.pack_forget()
        self.mode_frame.pack(fill=tk.X, padx=10, pady=5)
        self.api_frame.pack(fill=tk.X, padx=10, pady=5)
        self.settings_frame.pack(fill=tk.X, padx=10, pady=5)
        self.action_frame.pack(fill=tk.X, padx=10, pady=5)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def browse_file(self):
        """浏览选择视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("MP4文件", "*.mp4"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.video_file.set(file_path)
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def browse_directory(self):
        """浏览选择视频目录"""
        dir_path = filedialog.askdirectory(title="选择包含视频文件的文件夹")
        if dir_path:
            self.video_dir.set(dir_path)
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, dir_path)
            self.scan_video_files(dir_path)

    def scan_video_files(self, directory):
        """扫描目录中的视频文件"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
        video_files = []

        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(os.path.join(root, file))

            # 显示找到的文件
            self.file_list_text.delete(1.0, tk.END)
            if video_files:
                self.file_list_text.insert(tk.END, f"找到 {len(video_files)} 个视频文件:\n\n")
                for i, file_path in enumerate(video_files[:20], 1):  # 最多显示20个
                    rel_path = os.path.relpath(file_path, directory)
                    self.file_list_text.insert(tk.END, f"{i}. {rel_path}\n")

                if len(video_files) > 20:
                    self.file_list_text.insert(tk.END, f"\n... 还有 {len(video_files) - 20} 个文件")

                self.log_message(f"扫描完成，找到 {len(video_files)} 个视频文件")
            else:
                self.file_list_text.insert(tk.END, "该目录下没有找到视频文件")
                self.log_message("未找到视频文件")

            return video_files

        except Exception as e:
            self.log_message(f"扫描目录失败: {e}")
            self.file_list_text.delete(1.0, tk.END)
            self.file_list_text.insert(tk.END, f"扫描失败: {e}")
            return []

    def on_processing_mode_change(self):
        """处理模式切换回调"""
        mode = self.processing_mode.get()

        if mode == "single":
            # 单文件模式
            self.browse_btn.config(state=tk.NORMAL)
            self.browse_dir_btn.config(state=tk.DISABLED)
            self.file_list_frame.pack_forget()
            self.file_entry.delete(0, tk.END)
            if self.video_file.get():
                self.file_entry.insert(0, self.video_file.get())
        else:
            # 批量模式
            self.browse_btn.config(state=tk.DISABLED)
            self.browse_dir_btn.config(state=tk.NORMAL)
            self.file_list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            self.file_list_label.pack(anchor=tk.W)
            self.file_list_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.file_entry.delete(0, tk.END)
            if self.video_dir.get():
                self.file_entry.insert(0, self.video_dir.get())
                self.scan_video_files(self.video_dir.get())

    def on_mode_change(self):
        """模式切换回调"""
        mode = self.mode_var.get()

        # 模式标签映射
        mode_labels = {
            "qwen": "通义千问API密钥:",
            "gemini": "Gemini API密钥:",
            "deepseek": "DeepSeek API密钥:",
            "kimi": "月之暗面(Kimi) API密钥:",
            "zhipu": "智谱AI API密钥:",
            "baichuan": "百川智能API密钥:",
            "zeroone": "01.AI API密钥:",
            "openai": "OpenAI API密钥:",
            "custom": "自定义服务API密钥:",
            "simulate": "模拟模式无需API密钥"
        }

        # 更新API标签
        label_text = mode_labels.get(mode, "API密钥:")
        self.api_label.config(text=label_text)

        # 重置连接状态
        self.connection_status_var.set("未测试")
        self.connection_status_label.config(foreground="gray")

        if mode == "simulate":
            # 模拟模式
            self.api_entry.config(state=tk.DISABLED)
            self.test_api_btn.config(state=tk.DISABLED)
            self.custom_config_frame.pack_forget()
        elif mode == "custom":
            # 自定义模式需要显示额外配置
            self.api_entry.config(state=tk.NORMAL)
            self.test_api_btn.config(state=tk.NORMAL)
            self.custom_config_frame.pack(fill=tk.X, pady=(5, 0))

            # 设置默认值
            if not self.base_url_var.get():
                self.base_url_var.set("https://api.your-service.com")
            if not self.model_var.get():
                self.model_var.set("dall-e-3")

        else:
            # 其他模式
            self.api_entry.config(state=tk.NORMAL)
            self.test_api_btn.config(state=tk.NORMAL)
            self.custom_config_frame.pack_forget()

            # 为OpenAI兼容模式设置默认URL和模型
            mode_defaults = {
                "deepseek": ("https://api.deepseek.com", "deepseek-vl"),
                "kimi": ("https://api.moonshot.cn", "moonshot-v1-vision"),
                "zhipu": ("https://open.bigmodel.cn", "cogview-3"),
                "baichuan": ("https://api.baichuan-ai.com", "baichuan2-turbo"),
                "zeroone": ("https://api.lingyiwanwu.com", "yi-vision"),
                "openai": ("https://api.openai.com", "dall-e-3"),
            }

            if mode in mode_defaults:
                base_url, model = mode_defaults[mode]
                self.base_url_var.set(base_url)
                self.model_var.set(model)

    def update_threshold_label(self, value):
        """更新搞笑阈值标签"""
        self.threshold_label.config(text=f"{float(value):.2f}")

    def update_volume_label(self, value):
        """更新音量标签"""
        self.volume_label.config(text=f"{float(value):.2f}")

    def test_api(self):
        """测试API连接"""
        mode = self.mode_var.get()
        api_key = self.api_key_var.get().strip()

        if mode == "simulate":
            self.connection_status_var.set("✅ 模拟模式总是可用")
            self.connection_status_label.config(foreground="green")
            self.log_message("模拟模式无需测试API")
            return

        if not api_key:
            messagebox.showwarning("警告", "请先输入API密钥")
            return

        # 更新状态显示
        self.connection_status_var.set("🔄 测试中...")
        self.connection_status_label.config(foreground="orange")
        self.test_api_btn.config(state=tk.DISABLED)

        def test_api_thread():
            try:
                from universal_image_editor import UniversalImageEditor

                # 准备参数
                kwargs = {'api_key': api_key}

                if mode == "custom":
                    base_url = self.base_url_var.get().strip()
                    model = self.model_var.get().strip()

                    if not base_url:
                        self.message_queue.put(("api_test_result", False, "请输入Base URL"))
                        return

                    kwargs['base_url'] = base_url
                    kwargs['model'] = model

                self.log_message(f"正在测试{mode}API连接...")

                # 创建编辑器并测试连接
                editor = UniversalImageEditor(mode=mode, **kwargs)
                success = editor.test_connection()

                if success:
                    self.message_queue.put(("api_test_result", True, f"✅ {mode}API连接成功！"))
                else:
                    self.message_queue.put(("api_test_result", False, f"❌ {mode}API连接失败"))

            except Exception as e:
                self.message_queue.put(("api_test_result", False, f"❌ API测试异常: {e}"))

        # 在新线程中测试
        thread = threading.Thread(target=test_api_thread, daemon=True)
        thread.start()

    def start_processing(self):
        """开始处理"""
        processing_mode = self.processing_mode.get()

        # 验证输入
        if processing_mode == "single":
            if not self.video_file.get():
                messagebox.showwarning("警告", "请选择视频文件")
                return

            if not os.path.exists(self.video_file.get()):
                messagebox.showerror("错误", "视频文件不存在")
                return
        else:
            if not self.video_dir.get():
                messagebox.showwarning("警告", "请选择视频文件夹")
                return

            if not os.path.exists(self.video_dir.get()):
                messagebox.showerror("错误", "选择的文件夹不存在")
                return

            # 检查文件夹中是否有视频文件
            video_files = self.scan_video_files(self.video_dir.get())
            if not video_files:
                messagebox.showwarning("警告", "选择的文件夹中没有找到视频文件")
                return

        mode = self.mode_var.get()
        if mode in ["qwen", "gemini"] and not self.api_key_var.get().strip():
            messagebox.showwarning("警告", f"请输入{mode.upper()}API密钥")
            return

        # 批量处理确认
        if processing_mode == "batch":
            video_files = self.scan_video_files(self.video_dir.get())
            result = messagebox.askyesno("确认批量处理",
                                       f"找到 {len(video_files)} 个视频文件，确定要批量处理吗？\n\n"
                                       f"预估时间: {len(video_files) * 3} 分钟\n"
                                       f"这可能需要较长时间，建议在空闲时进行。")
            if not result:
                return

        # 保存设置
        self.save_config()

        # 更新界面状态
        self.is_processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)

        if processing_mode == "single":
            self.status_var.set("开始处理...")
            self.log_message("🚀 开始处理视频...")
        else:
            video_count = len(self.scan_video_files(self.video_dir.get()))
            self.status_var.set(f"开始批量处理 {video_count} 个视频...")
            self.log_message(f"🚀 开始批量处理 {video_count} 个视频...")

        # 清空日志
        self.log_text.delete(1.0, tk.END)

        # 在新线程中执行处理
        thread = threading.Thread(target=self.processing_thread, daemon=True)
        thread.start()

    def processing_thread(self):
        """处理线程"""
        try:
            # 更新配置文件
            self.update_config_for_processing()

            processing_mode = self.processing_mode.get()

            if processing_mode == "single":
                # 单文件处理
                self.process_single_video()
            else:
                # 批量处理
                self.process_batch_videos()

        except Exception as e:
            self.message_queue.put(("error", f"❌ 处理过程中出错: {e}"))
        finally:
            self.message_queue.put(("finished", None))

    def process_single_video(self):
        """处理单个视频"""
        try:
            def progress_callback(percentage, message):
                self.message_queue.put(("progress", percentage, message))

            workflow = GUIWorkflow(progress_callback=progress_callback)
            result = workflow.process_video(self.video_file.get())

            if result:
                self.message_queue.put(("success", f"✅ 视频处理完成！\n输出文件: {result}"))
            else:
                self.message_queue.put(("error", "❌ 视频处理失败"))

        except Exception as e:
            self.message_queue.put(("error", f"❌ 单文件处理失败: {e}"))

    def process_batch_videos(self):
        """批量处理视频"""
        try:
            video_files = self.scan_video_files(self.video_dir.get())
            total_videos = len(video_files)
            successful_count = 0
            failed_count = 0
            failed_files = []

            self.message_queue.put(("log", f"📂 开始批量处理 {total_videos} 个视频文件"))

            # 显示批量统计信息框架并初始化统计
            self.batch_stats_frame.pack(fill=tk.X, pady=(5, 0))
            self.message_queue.put(("batch_stats", total_videos, 0, 0, 0, []))

            for i, video_path in enumerate(video_files):
                if not self.is_processing:  # 检查是否被用户停止
                    self.message_queue.put(("log", "⏹️ 用户停止了批量处理"))
                    break

                try:
                    # 更新总体进度
                    overall_progress = int((i / total_videos) * 100)
                    video_name = os.path.basename(video_path)
                    self.message_queue.put(("progress", overall_progress,
                                          f"处理第 {i+1}/{total_videos} 个视频: {video_name}"))

                    self.message_queue.put(("log", f"🎬 [{i+1}/{total_videos}] 开始处理: {video_name}"))

                    # 创建单独的进度回调，调整到当前视频的进度范围
                    def video_progress_callback(percentage, message):
                        # 将单个视频的进度映射到总体进度
                        video_start = int((i / total_videos) * 100)
                        video_end = int(((i + 1) / total_videos) * 100)
                        adjusted_progress = video_start + int((percentage / 100) * (video_end - video_start))

                        self.message_queue.put(("progress", adjusted_progress,
                                              f"[{i+1}/{total_videos}] {video_name}: {message}"))

                    workflow = GUIWorkflow(progress_callback=video_progress_callback)
                    result = workflow.process_video(video_path)

                    if result:
                        successful_count += 1
                        self.message_queue.put(("log", f"✅ [{i+1}/{total_videos}] 处理成功: {video_name}"))
                        self.message_queue.put(("log", f"   输出文件: {os.path.basename(result)}"))
                    else:
                        failed_count += 1
                        failed_files.append(video_name)
                        self.message_queue.put(("log", f"❌ [{i+1}/{total_videos}] 处理失败: {video_name}"))

                    # 更新批量统计信息
                    self.message_queue.put(("batch_stats", total_videos, i+1, successful_count, failed_count, failed_files))

                except Exception as e:
                    failed_count += 1
                    failed_files.append(video_name)
                    self.message_queue.put(("log", f"❌ [{i+1}/{total_videos}] 处理异常: {video_name} - {e}"))

                    # 更新批量统计信息（异常情况）
                    self.message_queue.put(("batch_stats", total_videos, i+1, successful_count, failed_count, failed_files))

            # 批量处理完成总结
            self.message_queue.put(("progress", 100, "批量处理完成"))

            summary = f"🎊 批量处理完成！\n"
            summary += f"总共: {total_videos} 个视频\n"
            summary += f"成功: {successful_count} 个\n"
            summary += f"失败: {failed_count} 个"

            if failed_files:
                summary += f"\n\n失败的文件:\n" + "\n".join(f"• {f}" for f in failed_files[:10])
                if len(failed_files) > 10:
                    summary += f"\n... 还有 {len(failed_files) - 10} 个"

            if successful_count > 0:
                self.message_queue.put(("success", summary))
            else:
                self.message_queue.put(("error", summary))

        except Exception as e:
            self.message_queue.put(("error", f"❌ 批量处理失败: {e}"))

    def update_config_for_processing(self):
        """为处理更新配置文件"""
        try:
            # 根据选择的模式更新配置
            mode = self.mode_var.get()

            # 首先禁用所有模式
            all_modes = ["qwen", "deepseek", "kimi", "zhipu", "baichuan", "zeroone", "openai", "custom"]
            for m in all_modes:
                if self.config.has_section(m):
                    self.config.set(m, "enabled", "false")

            # 禁用传统模式
            if self.config.has_section("nano_banana"):
                self.config.set("nano_banana", "enable_real_generation", "false")
            if self.config.has_section("google_vision"):
                self.config.set("google_vision", "use_online_api", "false")

            # 启用选择的模式
            if mode == "qwen":
                self.config.set("qwen", "enabled", "true")
                self.config.set("qwen", "api_key", self.api_key_var.get())
            elif mode == "gemini":
                if not self.config.has_section("nano_banana"):
                    self.config.add_section("nano_banana")
                self.config.set("nano_banana", "enable_real_generation", "true")
                if not self.config.has_section("google_vision"):
                    self.config.add_section("google_vision")
                self.config.set("google_vision", "use_online_api", "true")
                self.config.set("google_vision", "api_key", self.api_key_var.get())
            elif mode == "simulate":
                # 模拟模式不需要特殊配置
                pass
            elif mode in all_modes:
                # OpenAI兼容模式
                if not self.config.has_section(mode):
                    self.config.add_section(mode)

                self.config.set(mode, "enabled", "true")
                self.config.set(mode, "api_key", self.api_key_var.get())

                # 设置base_url和model
                if mode == "custom":
                    self.config.set(mode, "base_url", self.base_url_var.get())
                    self.config.set(mode, "model", self.model_var.get())
                else:
                    # 使用默认值
                    mode_defaults = {
                        "deepseek": ("https://api.deepseek.com", "deepseek-vl"),
                        "kimi": ("https://api.moonshot.cn", "moonshot-v1-vision"),
                        "zhipu": ("https://open.bigmodel.cn", "cogview-3"),
                        "baichuan": ("https://api.baichuan-ai.com", "baichuan2-turbo"),
                        "zeroone": ("https://api.lingyiwanwu.com", "yi-vision"),
                        "openai": ("https://api.openai.com", "dall-e-3"),
                    }
                    if mode in mode_defaults:
                        base_url, model = mode_defaults[mode]
                        self.config.set(mode, "base_url", base_url)
                        self.config.set(mode, "model", model)

            # 更新其他设置
            self.config.set("video", "frame_interval", str(self.frame_interval_var.get()))
            if self.config.has_section("google_vision"):
                self.config.set("google_vision", "funny_score_threshold", str(self.threshold_var.get()))
                # 更新图片控制参数
                self.config.set("google_vision", "max_images", str(self.max_images_var.get()))
                self.config.set("google_vision", "min_interval_seconds", str(self.min_interval_var.get()))
                self.config.set("google_vision", "images_per_minute", str(self.images_per_minute_var.get()))
                self.config.set("google_vision", "auto_limit_mode", "smart" if self.smart_limit_var.get() else "manual")
            self.config.set("output", "bgm_volume", str(self.volume_var.get()))

            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)

        except Exception as e:
            self.message_queue.put(("error", f"更新配置失败: {e}"))

    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("已停止")
        self.log_message("⏹️ 用户停止了处理")

    def open_output_dir(self):
        """打开输出目录"""
        output_dir = "./drafts"

        # 确保目录存在
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.log_message(f"创建输出目录失败: {e}")
            messagebox.showerror("错误", f"无法创建输出目录: {e}")
            return

        # 打开目录
        try:
            # 转换为绝对路径
            abs_path = os.path.abspath(output_dir)
            os.startfile(abs_path)
            self.log_message(f"已打开输出目录: {abs_path}")
        except Exception as e:
            self.log_message(f"打开输出目录失败: {e}")
            messagebox.showerror("错误", f"无法打开输出目录: {e}\n路径: {os.path.abspath(output_dir)}")

    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"

        def update_log():
            self.log_text.insert(tk.END, full_message)
            self.log_text.see(tk.END)
            # 保持日志窗口最多1000行
            lines = int(self.log_text.index('end-1c').split('.')[0])
            if lines > 1000:
                self.log_text.delete(1.0, "100.0")

        # 如果在主线程中调用，直接更新；否则通过消息队列
        try:
            update_log()
        except:
            self.message_queue.put(("log", message))

    def update_batch_stats(self, total, current, successful, failed, failed_files=None):
        """更新批量处理统计信息"""
        def update_stats():
            self.batch_stats_text.config(state=tk.NORMAL)
            self.batch_stats_text.delete(1.0, tk.END)

            stats_text = f"总计视频: {total} | 当前处理: {current}/{total} | 成功: {successful} | 失败: {failed}"
            if failed_files:
                stats_text += f"\n失败文件: {', '.join(failed_files[-3:])}"  # 只显示最近3个失败文件
                if len(failed_files) > 3:
                    stats_text += f" (还有{len(failed_files)-3}个)"

            self.batch_stats_text.insert(1.0, stats_text)
            self.batch_stats_text.config(state=tk.DISABLED)

        # 如果在主线程中调用，直接更新；否则通过消息队列
        try:
            update_stats()
        except:
            self.message_queue.put(("batch_stats", total, current, successful, failed, failed_files))

    def process_messages(self):
        """处理消息队列"""
        try:
            while True:
                message_type, *args = self.message_queue.get_nowait()

                if message_type == "log":
                    self.log_message(args[0])
                elif message_type == "progress":
                    self.progress_var.set(args[0])
                    if len(args) > 1:
                        self.status_var.set(args[1])
                elif message_type == "success":
                    self.log_message(args[0])
                    messagebox.showinfo("成功", args[0])
                elif message_type == "error":
                    self.log_message(args[0])
                    messagebox.showerror("错误", args[0])
                elif message_type == "batch_stats":
                    self.update_batch_stats(*args)
                elif message_type == "api_test_result":
                    success, message = args[0], args[1]
                    self.test_api_btn.config(state=tk.NORMAL)
                    if success:
                        self.connection_status_var.set(message)
                        self.connection_status_label.config(foreground="green")
                    else:
                        self.connection_status_var.set(message)
                        self.connection_status_label.config(foreground="red")
                elif message_type == "finished":
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.is_processing = False
                    # 隐藏批量统计信息
                    self.batch_stats_frame.pack_forget()

        except queue.Empty:
            pass

        # 继续处理消息
        self.root.after(100, self.process_messages)

    def run(self):
        """运行GUI"""
        self.root.mainloop()

def main():
    """主函数"""
    app = SoulArtistGUI()
    app.run()

if __name__ == "__main__":
    main()