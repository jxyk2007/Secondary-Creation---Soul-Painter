# -*- coding: utf-8 -*-
"""
OpenAI兼容模式图像编辑器
支持所有主流AI厂商的OpenAI兼容接口
"""
import os
import requests
import base64
import json
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class OpenAICompatibleImageEditor(ABC):
    """OpenAI兼容接口基类"""

    def __init__(self, api_key: str, base_url: str, model: str = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()

        if not self.api_key:
            raise ValueError("API密钥不能为空")

    def get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'SoulArtist/1.0'
        }

    def image_to_base64(self, image_path: str) -> str:
        """将本地图片转换为base64编码"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')
                return base64_str
        except Exception as e:
            logger.error(f"图片转base64失败: {e}")
            return ""

    def save_base64_image(self, base64_data: str, output_path: str, target_size: tuple = None) -> bool:
        """将base64图片数据保存到文件，可选择调整尺寸"""
        try:
            # 移除data:image前缀（如果有）
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]

            image_data = base64.b64decode(base64_data)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 如果需要调整尺寸
            if target_size:
                import cv2
                import numpy as np

                # 将bytes转换为numpy数组
                nparr = np.frombuffer(image_data, np.uint8)
                # 解码图像
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # 调整尺寸
                    target_width, target_height = target_size
                    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

                    # 保存调整后的图像
                    cv2.imwrite(output_path, resized_img)
                    logger.info(f"图片已保存并调整尺寸到 {target_width}x{target_height}: {output_path}")
                    return True
                else:
                    logger.error("无法解码图像数据")
                    return False
            else:
                # 直接保存原图
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"图片已保存: {output_path}")
                return True

        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            return False

    @abstractmethod
    def get_endpoint_url(self, endpoint_type: str) -> str:
        """获取特定端点的URL"""
        pass

    @abstractmethod
    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        """格式化请求数据"""
        pass

    @abstractmethod
    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """解析响应数据，返回图片base64"""
        pass

    def parse_response_url(self, response_data: Dict[str, Any]) -> Optional[str]:
        """解析URL格式的响应，下载图片并转换为base64"""
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                url = response_data["data"][0].get("url")
                if url:
                    logger.info(f"从URL下载图片: {url}")
                    # 下载图片
                    response = self.session.get(url, timeout=60)
                    if response.status_code == 200:
                        # 转换为base64
                        import base64
                        base64_str = base64.b64encode(response.content).decode('utf-8')
                        logger.info("成功下载并转换图片为base64")
                        return base64_str
                    else:
                        logger.error(f"下载图片失败: {response.status_code}")
                        return None
        except Exception as e:
            logger.error(f"解析URL响应失败: {e}")
        return None

    def test_connection(self) -> bool:
        """测试API连接 - 优先测试edits接口"""
        try:
            logger.info(f"正在测试{self.__class__.__name__}连接...")

            # 创建一个简单的测试图片(1x1像素的红色图片)
            import io
            from PIL import Image

            test_image = Image.new('RGB', (1, 1), color='red')
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            test_prompt = "test connection"
            endpoint_url = self.get_endpoint_url("edit")

            # 尝试edits接口(multipart/form-data)
            files = {
                'image': ('test.png', img_byte_arr, 'image/png')
            }

            data = {
                'prompt': test_prompt,
                'n': '1',
                'size': '256x256'
            }

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'SoulArtist/1.0'
            }

            response = self.session.post(endpoint_url, headers=headers, data=data, files=files, timeout=60)

            if response.status_code == 200:
                logger.info(f"{self.__class__.__name__}连接测试成功(/v1/images/edits)")
                return True
            elif response.status_code == 404 or response.status_code == 405:
                # edits接口不存在,尝试generation接口
                logger.info(f"edits接口不可用,尝试generation接口")
                return self._test_generation_endpoint()
            else:
                logger.error(f"{self.__class__.__name__}连接测试失败: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"{self.__class__.__name__}连接测试异常: {e}")
            return False

    def _test_generation_endpoint(self) -> bool:
        """测试generation接口"""
        try:
            test_prompt = "A simple red circle"

            request_data = self.format_request_data(
                prompt=test_prompt,
                size="256x256",
                n=1
            )

            # 尝试url格式
            request_data['response_format'] = 'url'

            endpoint_url = self.get_endpoint_url("generation")
            headers = self.get_headers()

            response = self.session.post(endpoint_url, headers=headers, json=request_data, timeout=30)

            if response.status_code == 200:
                logger.info(f"{self.__class__.__name__}连接测试成功(/v1/images/generations)")
                return True
            else:
                logger.error(f"generation接口测试失败: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"generation接口测试异常: {e}")
            return False

    def convert_to_hand_drawn(self, image_path: str, prompt: str = None) -> Optional[str]:
        """将图片转换为手绘风格"""
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            return None

        # 默认手绘风格提示词
        if prompt is None:
            prompt = ("Convert this image to hand-drawn cartoon style. Requirements: "
                     "1) Keep main elements and composition from original image "
                     "2) Draw with black and white lines, with hand-drawn feel "
                     "3) Appropriately exaggerate facial expressions and actions for comedy "
                     "4) Add comic-style effect lines and emotion symbols "
                     "5) Overall style should look like hand-drawn sketchbook")

        try:
            endpoint_url = self.get_endpoint_url("edit")

            # 尝试使用multipart/form-data格式(标准OpenAI /v1/images/edits)
            logger.info(f"正在使用{self.__class__.__name__}转换图片为手绘风格: {image_path}")

            # 准备multipart/form-data
            with open(image_path, 'rb') as f:
                files = {
                    'image': (os.path.basename(image_path), f, 'image/png')
                }

                data = {
                    'prompt': prompt,
                    'n': '1',
                    'size': '1024x1024'
                }

                # 不使用Content-Type: application/json的headers
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'User-Agent': 'SoulArtist/1.0'
                }

                response = self.session.post(endpoint_url, headers=headers, data=data, files=files, timeout=120)

            if response.status_code == 200:
                result = response.json()
                # 优先尝试解析url格式
                parsed_result = self.parse_response_url(result)
                if parsed_result:
                    return parsed_result
                # 如果url格式失败,尝试b64_json格式
                return self.parse_response(result)
            elif response.status_code == 404 or response.status_code == 405:
                # 如果edits接口不存在,fallback到generations接口
                logger.warning(f"{self.__class__.__name__}不支持/edits接口,尝试使用/generations接口")
                return self._fallback_to_generation(image_path, prompt)
            else:
                logger.error(f"图片转换失败: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"图片转换异常: {e}")
            return None

    def _fallback_to_generation(self, image_path: str, prompt: str) -> Optional[str]:
        """fallback到generation接口(适用于不支持edits的服务)"""
        try:
            # 将本地图片转为base64
            image_base64 = self.image_to_base64(image_path)
            if not image_base64:
                return None

            # 构建请求数据
            request_data = self.format_request_data(
                prompt=prompt,
                image_base64=image_base64
            )

            endpoint_url = self.get_endpoint_url("generation")
            headers = self.get_headers()

            logger.info(f"使用generation接口作为备用方案")
            response = self.session.post(endpoint_url, headers=headers, json=request_data, timeout=120)

            if response.status_code == 200:
                result = response.json()
                return self.parse_response(result)
            else:
                logger.error(f"generation接口也失败: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"fallback异常: {e}")
            return None


class DeepSeekImageEditor(OpenAICompatibleImageEditor):
    """DeepSeek图像编辑器"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model="deepseek-vl"
        )

    def get_endpoint_url(self, endpoint_type: str) -> str:
        if endpoint_type == "generation":
            return f"{self.base_url}/v1/images/generations"
        elif endpoint_type == "edit":
            return f"{self.base_url}/v1/images/edits"
        else:
            return f"{self.base_url}/v1/images/generations"

    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "response_format": "b64_json"
        }

        if image_base64:
            data["image"] = image_base64

        return data

    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                return response_data["data"][0].get("b64_json")
        except Exception as e:
            logger.error(f"解析DeepSeek响应失败: {e}")
        return None


class KimiImageEditor(OpenAICompatibleImageEditor):
    """月之暗面(Kimi)图像编辑器"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.moonshot.cn",
            model="moonshot-v1-vision"
        )

    def get_endpoint_url(self, endpoint_type: str) -> str:
        if endpoint_type == "generation":
            return f"{self.base_url}/v1/images/generations"
        elif endpoint_type == "edit":
            return f"{self.base_url}/v1/images/edits"
        else:
            return f"{self.base_url}/v1/images/generations"

    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "response_format": "b64_json"
        }

        if image_base64:
            data["image"] = image_base64

        return data

    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                return response_data["data"][0].get("b64_json")
        except Exception as e:
            logger.error(f"解析Kimi响应失败: {e}")
        return None


class ZhipuImageEditor(OpenAICompatibleImageEditor):
    """智谱AI图像编辑器"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://open.bigmodel.cn",
            model="cogview-3"
        )

    def get_endpoint_url(self, endpoint_type: str) -> str:
        if endpoint_type == "generation":
            return f"{self.base_url}/api/paas/v4/images/generations"
        elif endpoint_type == "edit":
            return f"{self.base_url}/api/paas/v4/images/edits"
        else:
            return f"{self.base_url}/api/paas/v4/images/generations"

    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "response_format": "b64_json"
        }

        if image_base64:
            data["image"] = image_base64

        return data

    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                return response_data["data"][0].get("b64_json")
        except Exception as e:
            logger.error(f"解析智谱AI响应失败: {e}")
        return None


class BaichuanImageEditor(OpenAICompatibleImageEditor):
    """百川智能图像编辑器"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.baichuan-ai.com",
            model="baichuan2-turbo"
        )

    def get_endpoint_url(self, endpoint_type: str) -> str:
        if endpoint_type == "generation":
            return f"{self.base_url}/v1/images/generations"
        elif endpoint_type == "edit":
            return f"{self.base_url}/v1/images/edits"
        else:
            return f"{self.base_url}/v1/images/generations"

    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "response_format": "b64_json"
        }

        if image_base64:
            data["image"] = image_base64

        return data

    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                return response_data["data"][0].get("b64_json")
        except Exception as e:
            logger.error(f"解析百川智能响应失败: {e}")
        return None


class ZeroOneImageEditor(OpenAICompatibleImageEditor):
    """01.AI(零一万物)图像编辑器"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.lingyiwanwu.com",
            model="yi-vision"
        )

    def get_endpoint_url(self, endpoint_type: str) -> str:
        if endpoint_type == "generation":
            return f"{self.base_url}/v1/images/generations"
        elif endpoint_type == "edit":
            return f"{self.base_url}/v1/images/edits"
        else:
            return f"{self.base_url}/v1/images/generations"

    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "response_format": "b64_json"
        }

        if image_base64:
            data["image"] = image_base64

        return data

    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                return response_data["data"][0].get("b64_json")
        except Exception as e:
            logger.error(f"解析01.AI响应失败: {e}")
        return None


class OpenAIImageEditor(OpenAICompatibleImageEditor):
    """OpenAI官方图像编辑器"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com",
            model="dall-e-3"
        )

    def get_endpoint_url(self, endpoint_type: str) -> str:
        if endpoint_type == "generation":
            return f"{self.base_url}/v1/images/generations"
        elif endpoint_type == "edit":
            return f"{self.base_url}/v1/images/edits"
        else:
            return f"{self.base_url}/v1/images/generations"

    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "response_format": "b64_json"
        }

        if image_base64:
            data["image"] = image_base64

        return data

    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                return response_data["data"][0].get("b64_json")
        except Exception as e:
            logger.error(f"解析OpenAI响应失败: {e}")
        return None


class CustomOpenAIImageEditor(OpenAICompatibleImageEditor):
    """自定义OpenAI兼容服务图像编辑器"""

    def __init__(self, api_key: str, base_url: str, model: str = "dall-e-3"):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model
        )

    def get_endpoint_url(self, endpoint_type: str) -> str:
        if endpoint_type == "generation":
            return f"{self.base_url}/v1/images/generations"
        elif endpoint_type == "edit":
            return f"{self.base_url}/v1/images/edits"
        else:
            return f"{self.base_url}/v1/images/generations"

    def format_request_data(self, prompt: str, image_base64: str = None, **kwargs) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "model": self.model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "response_format": "b64_json"
        }

        if image_base64:
            data["image"] = image_base64

        return data

    def parse_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        try:
            if "data" in response_data and len(response_data["data"]) > 0:
                return response_data["data"][0].get("b64_json")
        except Exception as e:
            logger.error(f"解析自定义OpenAI服务响应失败: {e}")
        return None


# 厂商注册表
SUPPORTED_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "class": DeepSeekImageEditor,
        "description": "DeepSeek AI图像生成"
    },
    "kimi": {
        "name": "月之暗面(Kimi)",
        "class": KimiImageEditor,
        "description": "月之暗面Kimi图像生成"
    },
    "zhipu": {
        "name": "智谱AI",
        "class": ZhipuImageEditor,
        "description": "智谱AI CogView图像生成"
    },
    "baichuan": {
        "name": "百川智能",
        "class": BaichuanImageEditor,
        "description": "百川智能图像生成"
    },
    "zeroone": {
        "name": "01.AI(零一万物)",
        "class": ZeroOneImageEditor,
        "description": "01.AI零一万物图像生成"
    },
    "openai": {
        "name": "OpenAI官方",
        "class": OpenAIImageEditor,
        "description": "OpenAI官方DALL-E图像生成"
    },
    "custom": {
        "name": "自定义OpenAI兼容",
        "class": CustomOpenAIImageEditor,
        "description": "自定义OpenAI兼容服务"
    }
}


def create_image_editor(provider: str, api_key: str, **kwargs) -> Optional[OpenAICompatibleImageEditor]:
    """创建图像编辑器实例"""
    if provider not in SUPPORTED_PROVIDERS:
        logger.error(f"不支持的提供商: {provider}")
        return None

    try:
        provider_info = SUPPORTED_PROVIDERS[provider]
        editor_class = provider_info["class"]

        if provider == "custom":
            # 自定义服务需要额外参数
            base_url = kwargs.get("base_url")
            model = kwargs.get("model", "dall-e-3")
            if not base_url:
                logger.error("自定义服务需要提供base_url参数")
                return None
            return editor_class(api_key=api_key, base_url=base_url, model=model)
        else:
            return editor_class(api_key=api_key)

    except Exception as e:
        logger.error(f"创建{provider}图像编辑器失败: {e}")
        return None


def test_all_providers():
    """测试所有支持的提供商（需要有效的API密钥）"""
    print("=== OpenAI兼容图像编辑器测试 ===\n")

    # 这里需要配置实际的API密钥进行测试
    test_configs = {
        # "deepseek": "your_deepseek_api_key",
        # "kimi": "your_kimi_api_key",
        # "zhipu": "your_zhipu_api_key",
        # "baichuan": "your_baichuan_api_key",
        # "zeroone": "your_zeroone_api_key",
        # "openai": "your_openai_api_key",
    }

    for provider, api_key in test_configs.items():
        if api_key and api_key != "your_" + provider + "_api_key":
            print(f"测试 {SUPPORTED_PROVIDERS[provider]['name']}...")
            editor = create_image_editor(provider, api_key)
            if editor and editor.test_connection():
                print(f"✅ {provider} 连接成功")
            else:
                print(f"❌ {provider} 连接失败")
        else:
            print(f"⏭️ 跳过 {provider} (未配置API密钥)")

    print("\n测试完成！")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 运行测试
    test_all_providers()