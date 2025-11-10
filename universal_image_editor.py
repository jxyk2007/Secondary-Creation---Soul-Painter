# -*- coding: utf-8 -*-
"""
通用图像编辑器工厂类
统一管理所有图像生成模式（Qwen、Gemini、OpenAI兼容、模拟）
"""
import os
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any
from qwen_image_editor import QwenImageEditor
from openai_compatible_image_editor import create_image_editor, SUPPORTED_PROVIDERS

logger = logging.getLogger(__name__)

class UniversalImageEditor:
    """通用图像编辑器"""

    def __init__(self, mode: str, **kwargs):
        """
        初始化通用图像编辑器

        Args:
            mode: 编辑模式 ('qwen', 'gemini', 'simulate', 或OpenAI兼容模式)
            **kwargs: 模式特定的参数
        """
        self.mode = mode
        self.editor = None
        self.kwargs = kwargs

        # 初始化对应的编辑器
        if mode == "qwen":
            self._init_qwen_editor()
        elif mode == "gemini":
            self._init_gemini_editor()
        elif mode == "simulate":
            self._init_simulate_editor()
        elif mode in SUPPORTED_PROVIDERS:
            self._init_openai_compatible_editor()
        else:
            raise ValueError(f"不支持的模式: {mode}")

    def _init_qwen_editor(self):
        """初始化通义千问编辑器"""
        try:
            api_key = self.kwargs.get('api_key')
            self.editor = QwenImageEditor(api_key=api_key)
            logger.info("通义千问图像编辑器初始化成功")
        except Exception as e:
            logger.error(f"通义千问图像编辑器初始化失败: {e}")
            raise

    def _init_gemini_editor(self):
        """初始化Gemini编辑器"""
        # 这里可以添加Gemini编辑器的实现
        # 目前保持现有的实现方式
        logger.info("Gemini模式保持现有实现")

    def _init_simulate_editor(self):
        """初始化模拟编辑器"""
        logger.info("模拟模式编辑器初始化成功")

    def _init_openai_compatible_editor(self):
        """初始化OpenAI兼容编辑器"""
        try:
            api_key = self.kwargs.get('api_key')
            base_url = self.kwargs.get('base_url', '')
            model = self.kwargs.get('model', '')

            if self.mode == "custom":
                self.editor = create_image_editor(
                    provider=self.mode,
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
            else:
                self.editor = create_image_editor(
                    provider=self.mode,
                    api_key=api_key
                )

            if self.editor:
                logger.info(f"{SUPPORTED_PROVIDERS[self.mode]['name']}图像编辑器初始化成功")
            else:
                raise Exception("编辑器创建失败")

        except Exception as e:
            logger.error(f"{self.mode}图像编辑器初始化失败: {e}")
            raise

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            if self.mode == "simulate":
                return True  # 模拟模式总是可用
            elif self.mode == "gemini":
                # 这里添加Gemini的连接测试
                return True  # 暂时返回True
            elif self.editor:
                return self.editor.test_connection()
            else:
                return False
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False

    def convert_to_hand_drawn(self, image_path: str, prompt: str = None) -> Optional[str]:
        """将图片转换为手绘风格"""
        try:
            if self.mode == "simulate":
                return self._simulate_hand_drawn(image_path)
            elif self.mode == "gemini":
                return self._gemini_hand_drawn(image_path, prompt)
            elif self.editor:
                return self.editor.convert_to_hand_drawn(image_path, prompt)
            else:
                logger.error("编辑器未初始化")
                return None

        except Exception as e:
            logger.error(f"图像转换失败: {e}")
            return None

    def _simulate_hand_drawn(self, image_path: str) -> Optional[str]:
        """模拟手绘效果"""
        try:
            import base64

            # 读取原始图像
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取图像: {image_path}")
                return None

            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 应用高斯模糊
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 边缘检测
            edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

            # 添加一些噪点来模拟手绘感
            noise = np.random.randint(0, 50, gray.shape, dtype=np.uint8)
            edges = cv2.subtract(edges, noise)

            # 转换回3通道
            hand_drawn = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # 编码为base64
            _, buffer = cv2.imencode('.jpg', hand_drawn)
            base64_str = base64.b64encode(buffer).decode('utf-8')

            logger.info(f"模拟手绘效果生成成功: {image_path}")
            return base64_str

        except Exception as e:
            logger.error(f"模拟手绘效果生成失败: {e}")
            return None

    def _gemini_hand_drawn(self, image_path: str, prompt: str = None) -> Optional[str]:
        """Gemini手绘效果（保持现有实现）"""
        # 这里可以集成现有的Gemini实现
        # 暂时返回模拟效果
        logger.info("使用模拟效果代替Gemini实现")
        return self._simulate_hand_drawn(image_path)

    def save_base64_image(self, base64_data: str, output_path: str, target_size: tuple = None) -> bool:
        """保存base64图像数据"""
        try:
            if self.mode == "simulate" or self.mode == "gemini":
                # 对于模拟和Gemini模式，使用简单的保存方法
                import base64

                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',')[1]

                image_data = base64.b64decode(base64_data)

                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                if target_size:
                    # 使用OpenCV调整尺寸
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        target_width, target_height = target_size
                        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(output_path, resized_img)
                        logger.info(f"图片已保存并调整尺寸到 {target_width}x{target_height}: {output_path}")
                        return True
                else:
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    logger.info(f"图片已保存: {output_path}")
                    return True

            elif self.editor and hasattr(self.editor, 'save_base64_image'):
                return self.editor.save_base64_image(base64_data, output_path, target_size)
            else:
                logger.error("无法保存图像：编辑器不支持该功能")
                return False

        except Exception as e:
            logger.error(f"保存图像失败: {e}")
            return False

    def get_mode_info(self) -> Dict[str, Any]:
        """获取当前模式信息"""
        if self.mode in ["qwen", "gemini", "simulate"]:
            mode_names = {
                "qwen": "通义千问",
                "gemini": "Google Gemini",
                "simulate": "模拟模式"
            }
            return {
                "mode": self.mode,
                "name": mode_names.get(self.mode, self.mode),
                "description": f"{mode_names.get(self.mode, self.mode)}图像生成"
            }
        elif self.mode in SUPPORTED_PROVIDERS:
            provider_info = SUPPORTED_PROVIDERS[self.mode]
            return {
                "mode": self.mode,
                "name": provider_info["name"],
                "description": provider_info["description"]
            }
        else:
            return {
                "mode": self.mode,
                "name": "未知模式",
                "description": "未知的图像生成模式"
            }


def get_all_supported_modes() -> Dict[str, Dict[str, Any]]:
    """获取所有支持的模式"""
    modes = {
        "qwen": {
            "name": "通义千问",
            "description": "阿里通义千问图像编辑",
            "type": "proprietary"
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "Google Gemini图像生成",
            "type": "proprietary"
        },
        "simulate": {
            "name": "模拟模式",
            "description": "OpenCV模拟手绘效果",
            "type": "local"
        }
    }

    # 添加OpenAI兼容模式
    for provider_key, provider_info in SUPPORTED_PROVIDERS.items():
        modes[provider_key] = {
            "name": provider_info["name"],
            "description": provider_info["description"],
            "type": "openai_compatible"
        }

    return modes


def test_universal_editor():
    """测试通用编辑器"""
    print("=== 通用图像编辑器测试 ===\n")

    # 测试模拟模式
    print("1. 测试模拟模式...")
    try:
        editor = UniversalImageEditor("simulate")
        if editor.test_connection():
            print("✅ 模拟模式初始化成功")
        else:
            print("❌ 模拟模式初始化失败")
    except Exception as e:
        print(f"❌ 模拟模式错误: {e}")

    # 获取所有支持的模式
    print("\n2. 支持的模式列表:")
    modes = get_all_supported_modes()
    for mode_key, mode_info in modes.items():
        print(f"   - {mode_key}: {mode_info['name']} ({mode_info['type']})")

    print("\n测试完成！")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 运行测试
    test_universal_editor()