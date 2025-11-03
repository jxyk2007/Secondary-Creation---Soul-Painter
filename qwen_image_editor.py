# -*- coding: utf-8 -*-
import os
import requests
import base64
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class QwenImageEditor:
    """通义千问图片编辑API封装"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('DASHSCOPE_API_KEY')
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置或api_key参数为空")

    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            # 使用官方示例进行测试
            test_data = {
                "model": "qwen-image-edit",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250923/znhvuj/shoes1.webp"
                                },
                                {
                                    "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250923/alubtv/shoes2.webp"
                                },
                                {
                                    "text": "用图中黄色的鞋替换图中白色的鞋"
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "negative_prompt": "",
                    "watermark": False
                }
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }

            logger.info("正在测试通义千问API连接...")
            response = requests.post(self.base_url, headers=headers, json=test_data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                logger.info("API连接测试成功")
                logger.debug(f"测试响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                return True
            else:
                logger.error(f"API连接测试失败: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"API连接测试异常: {e}")
            return False

    def image_to_base64(self, image_path: str) -> str:
        """将本地图片转换为base64编码"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_str}"
        except Exception as e:
            logger.error(f"图片转base64失败: {e}")
            return ""

    def convert_to_hand_drawn(self, image_path: str, prompt: str = None) -> Optional[str]:
        """将图片转换为手绘风格

        Args:
            image_path: 输入图片路径
            prompt: 转换提示词，如果为None则使用默认手绘风格提示

        Returns:
            生成的图片base64数据，失败返回None
        """
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            return None

        # 默认手绘风格提示词
        if prompt is None:
            prompt = ("请将这张图片转换成手绘漫画风格。要求：1）保持原图的主要元素和构图 "
                     "2）用黑白线条绘制，线条要有手绘感 3）适当夸张人物表情和动作，增加搞笑效果 "
                     "4）添加一些漫画风格的效果线和表情符号 5）整体风格要像手工绘制的速写本")

        try:
            # 将本地图片转为base64
            image_base64 = self.image_to_base64(image_path)
            if not image_base64:
                return None

            # 构建请求数据
            request_data = {
                "model": "qwen-image-edit",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": image_base64
                                },
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "negative_prompt": "low quality, blurry, distorted",
                    "watermark": False
                }
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }

            logger.info(f"正在转换图片为手绘风格: {image_path}")
            response = requests.post(self.base_url, headers=headers, json=request_data, timeout=120)

            if response.status_code == 200:
                result = response.json()

                # 解析响应获取生成的图片
                if 'output' in result and 'choices' in result['output']:
                    choices = result['output']['choices']
                    if choices and len(choices) > 0:
                        choice = choices[0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            if isinstance(content, list) and len(content) > 0:
                                for item in content:
                                    if isinstance(item, dict) and 'image' in item:
                                        image_url = item['image']
                                        # 下载生成的图片
                                        return self._download_image(image_url)

                logger.error("响应中没有找到生成的图片数据")
                return None

            else:
                logger.error(f"图片转换失败: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"图片转换异常: {e}")
            return None

    def _download_image(self, image_url: str) -> Optional[str]:
        """下载生成的图片并返回base64数据"""
        try:
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                image_data = response.content
                base64_str = base64.b64encode(image_data).decode('utf-8')
                return base64_str
            else:
                logger.error(f"下载图片失败: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"下载图片异常: {e}")
            return None

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

# 测试函数
def test_qwen_api():
    """测试通义千问API"""
    try:
        editor = QwenImageEditor()
        logger.info("开始测试通义千问图片编辑API...")

        # 测试连接
        if editor.test_connection():
            logger.info("✅ API连接测试成功")
            return True
        else:
            logger.error("❌ API连接测试失败")
            return False

    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 运行测试
    test_qwen_api()