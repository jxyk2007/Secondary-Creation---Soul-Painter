# -*- coding: utf-8 -*-
"""
灵魂画手 GUI 主界面 - 紧凑版本
使用选项卡和下拉菜单优化布局
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
        self.root.title("灵魂画手 v202510130929 - AI视频二创工具")
        self.root.geometry("800x550")  # 从700降低到550
        self.root.resizable(True, True)

        # 设置图标
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        # 配置管理
        self.config = configparser.ConfigParser()
        self.config_file = "config.ini"
        self.load_config()

        # 状态变量
        self.video_file = tk.StringVar()
        self.video_dir = tk.StringVar()
        self.processing_mode = tk.StringVar(value="single")

        # 使用下拉菜单替代单选按钮
        self.mode_var = tk.StringVar(value="qwen")
        self.api_key_var = tk.StringVar()
        self.frame_interval_var = tk.IntVar(value=5000)
        self.threshold_var = tk.DoubleVar(value=0.85)
        self.volume_var = tk.DoubleVar(value=0.5)

        # 进度相关
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="准备就绪")
        self.is_processing = False

        # 消息队列
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
            if not self.config.has_section("qwen"):
                self.config.add_section("qwen")
            self.config.set("qwen", "api_key", self.api_key_var.get())

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

            # 加载视频分析模式
            if self.config.has_option("video_analysis", "mode"):
                analysis_mode = self.config.get("video_analysis", "mode")
                display_name = self.analysis_mode_map_reverse.get(analysis_mode, "Ollama本地模型(推荐)")
                self.analysis_mode_var.set(display_name)

            # 加载Ollama模型
            if self.config.has_option("video_analysis", "model"):
                ollama_model = self.config.get("video_analysis", "model")
                self.ollama_model_var.set(ollama_model)

        except Exception as e:
            self.log_message(f"加载设置失败: {e}")

    def create_widgets(self):
        """创建界面组件 - 紧凑版"""

        # 主标题
        title_frame = ttk.Frame(self.root)
        title_label = ttk.Label(title_frame, text="🎨 灵魂画手 v1.0", font=("Microsoft YaHei", 14, "bold"))
        title_label.pack(pady=5)

        # 使用选项卡
        self.notebook = ttk.Notebook(self.root)

        # ===== 选项卡1: 基础设置 =====
        basic_tab = ttk.Frame(self.notebook)
        self.notebook.add(basic_tab, text="基础设置")

        # 文件选择(紧凑布局)
        file_frame = ttk.LabelFrame(basic_tab, text="📁 视频文件", padding=(5,5))
        file_frame.pack(fill=tk.X, padx=5, pady=3)

        # 第一行:处理模式
        mode_row = ttk.Frame(file_frame)
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_row, text="单个文件", variable=self.processing_mode,
                       value="single", command=self.on_processing_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_row, text="批量处理", variable=self.processing_mode,
                       value="batch", command=self.on_processing_mode_change).pack(side=tk.LEFT)

        # 第二行:文件选择
        file_row = ttk.Frame(file_frame)
        file_row.pack(fill=tk.X, pady=2)
        self.file_entry = ttk.Entry(file_row, width=50)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,3))
        self.browse_btn = ttk.Button(file_row, text="浏览", command=self.browse_file, width=8)
        self.browse_btn.pack(side=tk.LEFT, padx=1)
        self.browse_dir_btn = ttk.Button(file_row, text="文件夹", command=self.browse_directory,
                                        width=8, state=tk.DISABLED)
        self.browse_dir_btn.pack(side=tk.LEFT, padx=1)

        # AI模式选择(使用下拉菜单)
        ai_frame = ttk.LabelFrame(basic_tab, text="🤖 AI模式", padding=(5,5))
        ai_frame.pack(fill=tk.X, padx=5, pady=3)

        # 第一行:模式下拉菜单
        mode_row = ttk.Frame(ai_frame)
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="选择模式:").pack(side=tk.LEFT, padx=(0,5))

        mode_choices = [
            ("通义千问(推荐)", "qwen"),
            ("Gemini", "gemini"),
            ("DeepSeek", "deepseek"),
            ("月之暗面(Kimi)", "kimi"),
            ("智谱AI", "zhipu"),
            ("百川智能", "baichuan"),
            ("01.AI(零一万物)", "zeroone"),
            ("OpenAI官方", "openai"),
            ("自定义OpenAI兼容", "custom"),
            ("模拟模式(无需网络)", "simulate")
        ]

        self.mode_combo = ttk.Combobox(mode_row, textvariable=self.mode_var, width=25, state="readonly")
        self.mode_combo['values'] = [choice[0] for choice in mode_choices]
        self.mode_combo.current(0)
        self.mode_combo.pack(side=tk.LEFT, padx=(0,5))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_mode_change())

        # 存储映射
        self.mode_map = {choice[0]: choice[1] for choice in mode_choices}
        self.mode_map_reverse = {choice[1]: choice[0] for choice in mode_choices}

        # 第二行:API密钥
        api_row = ttk.Frame(ai_frame)
        api_row.pack(fill=tk.X, pady=2)
        self.api_label = ttk.Label(api_row, text="API密钥:", width=10)
        self.api_label.pack(side=tk.LEFT)
        self.api_entry = ttk.Entry(api_row, textvariable=self.api_key_var, width=40)
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,3))
        self.test_api_btn = ttk.Button(api_row, text="测试", command=self.test_api, width=8)
        self.test_api_btn.pack(side=tk.LEFT)

        # 第三行:连接状态
        status_row = ttk.Frame(ai_frame)
        status_row.pack(fill=tk.X, pady=2)
        ttk.Label(status_row, text="状态:").pack(side=tk.LEFT, padx=(0,5))
        self.connection_status_var = tk.StringVar(value="未测试")
        self.connection_status_label = ttk.Label(status_row, textvariable=self.connection_status_var,
                                                foreground="gray")
        self.connection_status_label.pack(side=tk.LEFT)

        # 自定义配置区域(默认隐藏)
        self.custom_config_frame = ttk.LabelFrame(ai_frame, text="自定义配置", padding=(5,3))

        custom_row1 = ttk.Frame(self.custom_config_frame)
        custom_row1.pack(fill=tk.X, pady=1)
        ttk.Label(custom_row1, text="Base URL:", width=10).pack(side=tk.LEFT)
        self.base_url_var = tk.StringVar()
        ttk.Entry(custom_row1, textvariable=self.base_url_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)

        custom_row2 = ttk.Frame(self.custom_config_frame)
        custom_row2.pack(fill=tk.X, pady=1)
        ttk.Label(custom_row2, text="模型名称:", width=10).pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        ttk.Entry(custom_row2, textvariable=self.model_var, width=30).pack(side=tk.LEFT)

        # ===== 选项卡2: 高级设置 =====
        advanced_tab = ttk.Frame(self.notebook)
        self.notebook.add(advanced_tab, text="高级设置")

        # 视频分析模式
        analysis_frame = ttk.LabelFrame(advanced_tab, text="🎥 视频分析模式", padding=(5,5))
        analysis_frame.pack(fill=tk.X, padx=5, pady=3)

        analysis_row = ttk.Frame(analysis_frame)
        analysis_row.pack(fill=tk.X, pady=2)
        ttk.Label(analysis_row, text="分析模式:", width=12).pack(side=tk.LEFT)

        self.analysis_mode_var = tk.StringVar(value="ollama")
        analysis_choices = [
            ("Ollama本地模型(推荐)", "ollama"),
            ("Google Gemini", "gemini"),
            ("通义千问视频", "qwen_video"),
            ("自定义逐帧API", "frame_by_frame"),
            ("模拟模式", "simulate")
        ]

        analysis_combo = ttk.Combobox(analysis_row, textvariable=self.analysis_mode_var,
                                     width=25, state="readonly")
        analysis_combo['values'] = [choice[0] for choice in analysis_choices]
        analysis_combo.current(0)
        analysis_combo.pack(side=tk.LEFT, padx=3)

        # 存储映射
        self.analysis_mode_map = {choice[0]: choice[1] for choice in analysis_choices}
        self.analysis_mode_map_reverse = {choice[1]: choice[0] for choice in analysis_choices}

        # Ollama模型选择
        ollama_row = ttk.Frame(analysis_frame)
        ollama_row.pack(fill=tk.X, pady=2)
        ttk.Label(ollama_row, text="Ollama模型:", width=12).pack(side=tk.LEFT)

        self.ollama_model_var = tk.StringVar(value="llava")
        ollama_model_combo = ttk.Combobox(ollama_row, textvariable=self.ollama_model_var,
                                         width=20, state="readonly")
        ollama_model_combo['values'] = ["llava", "llava:13b", "llava:34b", "bakllava",
                                        "llava-phi3", "llava-llama3"]
        ollama_model_combo.current(0)
        ollama_model_combo.pack(side=tk.LEFT, padx=3)

        ttk.Label(ollama_row, text="(需先安装: ollama pull llava)", foreground="gray").pack(side=tk.LEFT, padx=5)

        # 视频参数
        video_frame = ttk.LabelFrame(advanced_tab, text="视频参数", padding=(5,5))
        video_frame.pack(fill=tk.X, padx=5, pady=3)

        # 抽帧间隔
        interval_row = ttk.Frame(video_frame)
        interval_row.pack(fill=tk.X, pady=2)
        ttk.Label(interval_row, text="抽帧间隔:", width=12).pack(side=tk.LEFT)
        ttk.Spinbox(interval_row, from_=1000, to=10000, width=10,
                   textvariable=self.frame_interval_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(interval_row, text="毫秒").pack(side=tk.LEFT)

        # 搞笑阈值
        threshold_row = ttk.Frame(video_frame)
        threshold_row.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_row, text="搞笑阈值:", width=12).pack(side=tk.LEFT)
        threshold_scale = ttk.Scale(threshold_row, from_=0.1, to=1.0,
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self.threshold_label = ttk.Label(threshold_row, text="0.85", width=5)
        self.threshold_label.pack(side=tk.LEFT)
        threshold_scale.configure(command=self.update_threshold_label)

        # BGM音量
        volume_row = ttk.Frame(video_frame)
        volume_row.pack(fill=tk.X, pady=2)
        ttk.Label(volume_row, text="BGM音量:", width=12).pack(side=tk.LEFT)
        volume_scale = ttk.Scale(volume_row, from_=0.0, to=1.0,
                                variable=self.volume_var, orient=tk.HORIZONTAL)
        volume_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self.volume_label = ttk.Label(volume_row, text="0.5", width=5)
        self.volume_label.pack(side=tk.LEFT)
        volume_scale.configure(command=self.update_volume_label)

        # 图片控制
        image_frame = ttk.LabelFrame(advanced_tab, text="图片生成控制", padding=(5,5))
        image_frame.pack(fill=tk.X, padx=5, pady=3)

        # 第一行
        img_row1 = ttk.Frame(image_frame)
        img_row1.pack(fill=tk.X, pady=1)
        ttk.Label(img_row1, text="最大图片数:", width=12).pack(side=tk.LEFT)
        self.max_images_var = tk.IntVar(value=6)
        ttk.Spinbox(img_row1, from_=1, to=20, width=8,
                   textvariable=self.max_images_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(img_row1, text="张").pack(side=tk.LEFT)

        # 第二行
        img_row2 = ttk.Frame(image_frame)
        img_row2.pack(fill=tk.X, pady=1)
        ttk.Label(img_row2, text="最小间隔:", width=12).pack(side=tk.LEFT)
        self.min_interval_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(img_row2, from_=1.0, to=30.0, increment=0.5, width=8,
                   textvariable=self.min_interval_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(img_row2, text="秒").pack(side=tk.LEFT, padx=(0,10))

        ttk.Label(img_row2, text="每分钟:", width=10).pack(side=tk.LEFT)
        self.images_per_minute_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(img_row2, from_=0.5, to=10.0, increment=0.5, width=8,
                   textvariable=self.images_per_minute_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(img_row2, text="张").pack(side=tk.LEFT)

        # 第三行
        img_row3 = ttk.Frame(image_frame)
        img_row3.pack(fill=tk.X, pady=1)
        self.smart_limit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(img_row3, text="智能限制模式(自动调整密度)",
                       variable=self.smart_limit_var).pack(side=tk.LEFT)

        # ===== 回到主界面: 操作按钮区域 =====
        action_frame = ttk.Frame(self.root)

        self.start_btn = ttk.Button(action_frame, text="🎬 开始生成",
                                   command=self.start_processing, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=3)

        self.stop_btn = ttk.Button(action_frame, text="⏹️ 停止",
                                  command=self.stop_processing, state=tk.DISABLED, width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=3)

        self.open_output_btn = ttk.Button(action_frame, text="📁 打开输出",
                                         command=self.open_output_dir, width=12)
        self.open_output_btn.pack(side=tk.LEFT, padx=3)

        # 进度显示区域
        progress_frame = ttk.LabelFrame(self.root, text="📊 处理进度", padding=(5,5))

        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 3))

        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W)

        # 日志输出区域
        log_frame = ttk.LabelFrame(self.root, text="📝 日志", padding=(5,5))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, width=70)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 存储组件引用
        self.title_frame = title_frame
        self.action_frame = action_frame
        self.progress_frame = progress_frame
        self.log_frame = log_frame

    def setup_layout(self):
        """设置布局"""
        self.title_frame.pack(fill=tk.X, padx=5, pady=3)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
        self.action_frame.pack(fill=tk.X, padx=5, pady=3)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=3)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

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

            self.log_message(f"扫描完成,找到 {len(video_files)} 个视频文件")
            return video_files

        except Exception as e:
            self.log_message(f"扫描目录失败: {e}")
            return []

    def on_processing_mode_change(self):
        """处理模式切换回调"""
        mode = self.processing_mode.get()

        if mode == "single":
            self.browse_btn.config(state=tk.NORMAL)
            self.browse_dir_btn.config(state=tk.DISABLED)
            self.file_entry.delete(0, tk.END)
            if self.video_file.get():
                self.file_entry.insert(0, self.video_file.get())
        else:
            self.browse_btn.config(state=tk.DISABLED)
            self.browse_dir_btn.config(state=tk.NORMAL)
            self.file_entry.delete(0, tk.END)
            if self.video_dir.get():
                self.file_entry.insert(0, self.video_dir.get())

    def on_mode_change(self):
        """模式切换回调"""
        display_name = self.mode_combo.get()
        mode = self.mode_map.get(display_name, "qwen")
        self.mode_var.set(mode)

        # 更新API标签
        mode_labels = {
            "qwen": "通义千问API:",
            "gemini": "Gemini API:",
            "deepseek": "DeepSeek API:",
            "kimi": "Kimi API:",
            "zhipu": "智谱AI API:",
            "baichuan": "百川API:",
            "zeroone": "01.AI API:",
            "openai": "OpenAI API:",
            "custom": "自定义API:",
            "simulate": "无需API"
        }

        label_text = mode_labels.get(mode, "API密钥:")
        self.api_label.config(text=label_text)

        # 重置连接状态
        self.connection_status_var.set("未测试")
        self.connection_status_label.config(foreground="gray")

        if mode == "simulate":
            self.api_entry.config(state=tk.DISABLED)
            self.test_api_btn.config(state=tk.DISABLED)
            self.custom_config_frame.pack_forget()
        elif mode == "custom":
            self.api_entry.config(state=tk.NORMAL)
            self.test_api_btn.config(state=tk.NORMAL)
            self.custom_config_frame.pack(fill=tk.X, pady=(3,0))
            if not self.base_url_var.get():
                self.base_url_var.set("https://api.your-service.com")
            if not self.model_var.get():
                self.model_var.set("dall-e-3")
        else:
            self.api_entry.config(state=tk.NORMAL)
            self.test_api_btn.config(state=tk.NORMAL)
            self.custom_config_frame.pack_forget()

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

        self.connection_status_var.set("🔄 测试中...")
        self.connection_status_label.config(foreground="orange")
        self.test_api_btn.config(state=tk.DISABLED)

        def test_api_thread():
            try:
                from universal_image_editor import UniversalImageEditor

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

                editor = UniversalImageEditor(mode=mode, **kwargs)
                success = editor.test_connection()

                if success:
                    self.message_queue.put(("api_test_result", True, f"✅ {mode}API连接成功!"))
                else:
                    self.message_queue.put(("api_test_result", False, f"❌ {mode}API连接失败"))

            except Exception as e:
                self.message_queue.put(("api_test_result", False, f"❌ API测试异常: {e}"))

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
                                       f"找到 {len(video_files)} 个视频文件,确定要批量处理吗?\n\n"
                                       f"预估时间: {len(video_files) * 3} 分钟\n"
                                       f"这可能需要较长时间,建议在空闲时进行。")
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
            self.update_config_for_processing()

            processing_mode = self.processing_mode.get()

            if processing_mode == "single":
                self.process_single_video()
            else:
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
                self.message_queue.put(("success", f"✅ 视频处理完成!\n输出文件: {result}"))
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

            for i, video_path in enumerate(video_files):
                if not self.is_processing:
                    self.message_queue.put(("log", "⏹️ 用户停止了批量处理"))
                    break

                try:
                    overall_progress = int((i / total_videos) * 100)
                    video_name = os.path.basename(video_path)
                    self.message_queue.put(("progress", overall_progress,
                                          f"处理第 {i+1}/{total_videos} 个视频: {video_name}"))

                    self.message_queue.put(("log", f"🎬 [{i+1}/{total_videos}] 开始处理: {video_name}"))

                    def video_progress_callback(percentage, message):
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
                    else:
                        failed_count += 1
                        failed_files.append(video_name)
                        self.message_queue.put(("log", f"❌ [{i+1}/{total_videos}] 处理失败: {video_name}"))

                except Exception as e:
                    failed_count += 1
                    failed_files.append(video_name)
                    self.message_queue.put(("log", f"❌ [{i+1}/{total_videos}] 处理异常: {video_name} - {e}"))

            self.message_queue.put(("progress", 100, "批量处理完成"))

            summary = f"🎊 批量处理完成!\n"
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
            mode = self.mode_var.get()

            # 禁用所有模式
            all_modes = ["qwen", "deepseek", "kimi", "zhipu", "baichuan", "zeroone", "openai", "custom"]
            for m in all_modes:
                if self.config.has_section(m):
                    self.config.set(m, "enabled", "false")

            if self.config.has_section("nano_banana"):
                self.config.set("nano_banana", "enable_real_generation", "false")
            if self.config.has_section("google_vision"):
                self.config.set("google_vision", "use_online_api", "false")

            # 启用选择的模式
            if mode == "qwen":
                if not self.config.has_section("qwen"):
                    self.config.add_section("qwen")
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
                pass
            elif mode in all_modes:
                if not self.config.has_section(mode):
                    self.config.add_section(mode)

                self.config.set(mode, "enabled", "true")
                self.config.set(mode, "api_key", self.api_key_var.get())

                if mode == "custom":
                    self.config.set(mode, "base_url", self.base_url_var.get())
                    self.config.set(mode, "model", self.model_var.get())
                else:
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

            # 更新视频分析配置
            if not self.config.has_section("video_analysis"):
                self.config.add_section("video_analysis")

            # 获取选中的分析模式
            analysis_display = self.analysis_mode_var.get()
            analysis_mode = self.analysis_mode_map.get(analysis_display, "ollama")

            self.config.set("video_analysis", "mode", analysis_mode)
            self.config.set("video_analysis", "enabled", "true")

            if analysis_mode == "ollama":
                self.config.set("video_analysis", "base_url", "http://localhost:11434")
                self.config.set("video_analysis", "model", self.ollama_model_var.get())
                self.config.set("video_analysis", "api_key", "not-needed")
            elif analysis_mode == "gemini":
                # Gemini使用google_vision的配置
                if not self.config.has_section("google_vision"):
                    self.config.add_section("google_vision")
                self.config.set("google_vision", "use_online_api", "true")
                self.config.set("google_vision", "api_key", self.api_key_var.get())

            # 更新其他设置
            if not self.config.has_section("video"):
                self.config.add_section("video")
            self.config.set("video", "frame_interval", str(self.frame_interval_var.get()))

            if not self.config.has_section("google_vision"):
                self.config.add_section("google_vision")
            self.config.set("google_vision", "funny_score_threshold", str(self.threshold_var.get()))
            self.config.set("google_vision", "max_images", str(self.max_images_var.get()))
            self.config.set("google_vision", "min_interval_seconds", str(self.min_interval_var.get()))
            self.config.set("google_vision", "images_per_minute", str(self.images_per_minute_var.get()))
            self.config.set("google_vision", "auto_limit_mode", "smart" if self.smart_limit_var.get() else "manual")

            if not self.config.has_section("output"):
                self.config.add_section("output")
            self.config.set("output", "bgm_volume", str(self.volume_var.get()))

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

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.log_message(f"创建输出目录失败: {e}")
            messagebox.showerror("错误", f"无法创建输出目录: {e}")
            return

        try:
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
            lines = int(self.log_text.index('end-1c').split('.')[0])
            if lines > 1000:
                self.log_text.delete(1.0, "100.0")

        try:
            update_log()
        except:
            self.message_queue.put(("log", message))

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

        except queue.Empty:
            pass

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
