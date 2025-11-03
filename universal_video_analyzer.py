# -*- coding: utf-8 -*-
"""
通用视频分析器
支持多种视频理解API:Gemini、通义千问、自定义OpenAI兼容等
"""
import os
import logging
import base64
import requests
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VideoAnalysisResult:
    """视频分析结果"""
    def __init__(self):
        self.funny_moments: List[Dict] = []  # [{"start_time": 0, "end_time": 5, "funny_score": 0.9, "description": "..."}]
        self.overall_score: float = 0.5
        self.description: str = ""


class UniversalVideoAnalyzer:
    """通用视频分析器工厂类"""

    def __init__(self, mode: str, **kwargs):
        """
        初始化视频分析器

        Args:
            mode: 分析模式 ('gemini', 'qwen_video', 'frame_by_frame', 'ollama', 'simulate')
            **kwargs: 模式特定参数(api_key, base_url等)
        """
        self.mode = mode
        self.kwargs = kwargs
        self.analyzer = None

        if mode == "gemini":
            self._init_gemini_analyzer()
        elif mode == "qwen_video":
            self._init_qwen_video_analyzer()
        elif mode == "frame_by_frame":
            self._init_frame_analyzer()
        elif mode == "ollama":
            self._init_ollama_analyzer()
        elif mode == "simulate":
            self._init_simulate_analyzer()
        else:
            raise ValueError(f"不支持的视频分析模式: {mode}")

    def _init_gemini_analyzer(self):
        """初始化Gemini视频分析器"""
        try:
            api_key = self.kwargs.get('api_key')
            proxy_enabled = self.kwargs.get('proxy_enabled', False)
            proxy_host = self.kwargs.get('proxy_host', '127.0.0.1')
            proxy_port = self.kwargs.get('proxy_port', '1080')

            self.analyzer = GeminiVideoAnalyzer(
                api_key=api_key,
                proxy_enabled=proxy_enabled,
                proxy_host=proxy_host,
                proxy_port=proxy_port
            )
            logger.info("Gemini视频分析器初始化成功")
        except Exception as e:
            logger.error(f"Gemini视频分析器初始化失败: {e}")
            raise

    def _init_qwen_video_analyzer(self):
        """初始化通义千问视频分析器"""
        try:
            api_key = self.kwargs.get('api_key')
            self.analyzer = QwenVideoAnalyzer(api_key=api_key)
            logger.info("通义千问视频分析器初始化成功")
        except Exception as e:
            logger.error(f"通义千问视频分析器初始化失败: {e}")
            raise

    def _init_frame_analyzer(self):
        """初始化逐帧分析器(使用图像理解模型)"""
        try:
            api_key = self.kwargs.get('api_key')
            base_url = self.kwargs.get('base_url')
            model = self.kwargs.get('model')

            self.analyzer = FrameByFrameAnalyzer(
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            logger.info("逐帧分析器初始化成功")
        except Exception as e:
            logger.error(f"逐帧分析器初始化失败: {e}")
            raise

    def _init_ollama_analyzer(self):
        """初始化Ollama视频分析器"""
        try:
            base_url = self.kwargs.get('base_url', 'http://localhost:11434')
            model = self.kwargs.get('model', 'llava')

            self.analyzer = OllamaVideoAnalyzer(
                base_url=base_url,
                model=model
            )
            logger.info(f"Ollama视频分析器初始化成功(模型:{model})")
        except Exception as e:
            logger.error(f"Ollama视频分析器初始化失败: {e}")
            raise

    def _init_simulate_analyzer(self):
        """初始化模拟分析器"""
        self.analyzer = SimulateAnalyzer()
        logger.info("模拟分析器初始化成功")

    def analyze_video(self, video_path: str, frame_info: List[Dict], threshold: float = 0.85) -> VideoAnalysisResult:
        """分析视频内容"""
        if not self.analyzer:
            logger.error("分析器未初始化")
            return VideoAnalysisResult()

        return self.analyzer.analyze_video(video_path, frame_info, threshold)

    def test_connection(self) -> bool:
        """测试连接"""
        if not self.analyzer:
            return False
        return self.analyzer.test_connection()


class BaseVideoAnalyzer(ABC):
    """视频分析器基类"""

    @abstractmethod
    def analyze_video(self, video_path: str, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """分析视频内容"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """测试API连接"""
        pass


class GeminiVideoAnalyzer(BaseVideoAnalyzer):
    """Gemini视频分析器"""

    def __init__(self, api_key: str, proxy_enabled: bool = False, proxy_host: str = '127.0.0.1', proxy_port: str = '1080'):
        self.api_key = api_key
        self.session = requests.Session()

        # 设置代理
        if proxy_enabled:
            proxy_url = f"http://{proxy_host}:{proxy_port}"
            self.proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
            logger.info(f"使用代理: {proxy_url}")
        else:
            self.proxies = None

    def test_connection(self) -> bool:
        """测试Gemini连接"""
        try:
            # 简单测试生成内容
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": "test"}]}]}

            response = self.session.post(url, headers=headers, json=data, proxies=self.proxies, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Gemini连接测试失败: {e}")
            return False

    def analyze_video(self, video_path: str, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """使用Gemini分析视频"""
        try:
            # 上传视频
            file_uri = self._upload_video(video_path)
            if not file_uri:
                logger.warning("视频上传失败,使用模拟数据")
                return self._simulate_analysis(frame_info, threshold)

            # 分析视频
            analysis_data = self._generate_content(file_uri)
            if not analysis_data:
                logger.warning("视频分析失败,使用模拟数据")
                return self._simulate_analysis(frame_info, threshold)

            # 解析结果
            return self._parse_gemini_response(analysis_data, frame_info, threshold)

        except Exception as e:
            logger.error(f"Gemini视频分析失败: {e}")
            return self._simulate_analysis(frame_info, threshold)

    def _upload_video(self, video_path: str) -> Optional[str]:
        """上传视频到Gemini"""
        try:
            # 检查文件大小
            file_size = os.path.getsize(video_path)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                logger.error(f"视频文件太大: {file_size / 1024 / 1024:.2f}MB")
                return None

            logger.info(f"开始上传视频: {video_path}")

            # 1. 开始上传
            headers = {
                "x-goog-api-key": self.api_key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": "video/mp4",
                "Content-Type": "application/json"
            }

            data = {"file": {"display_name": os.path.basename(video_path)}}

            response = self.session.post(
                "https://generativelanguage.googleapis.com/upload/v1beta/files",
                headers=headers,
                json=data,
                proxies=self.proxies,
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"开始上传失败: {response.status_code}")
                return None

            upload_url = response.headers.get("x-goog-upload-url")

            # 2. 上传视频数据
            headers = {
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize"
            }

            with open(video_path, 'rb') as f:
                response = self.session.post(
                    upload_url,
                    headers=headers,
                    data=f,
                    proxies=self.proxies,
                    timeout=600
                )

            if response.status_code == 200:
                result = response.json()
                file_uri = result.get("file", {}).get("uri", "")
                logger.info(f"视频上传成功: {file_uri}")
                return file_uri
            else:
                logger.error(f"上传失败: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"视频上传异常: {e}")
            return None

    def _generate_content(self, file_uri: str) -> Optional[Dict]:
        """生成分析内容"""
        try:
            prompt = """请分析这个视频,找出最搞笑、最有趣的时间段。
以JSON格式返回结果:
{
  "funny_moments": [
    {"start_time": 0.0, "end_time": 5.0, "funny_score": 0.9, "description": "描述"},
    ...
  ]
}"""

            data = {
                "contents": [{
                    "parts": [
                        {"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 4096
                }
            }

            headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json"
            }

            response = self.session.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                headers=headers,
                json=data,
                proxies=self.proxies,
                timeout=120
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"生成内容失败: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"生成内容异常: {e}")
            return None

    def _parse_gemini_response(self, response_data: Dict, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """解析Gemini响应"""
        result = VideoAnalysisResult()

        try:
            import json
            import re

            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                text = response_data["candidates"][0]["content"]["parts"][0].get("text", "")

                # 提取JSON
                json_match = re.search(r'\{[\s\S]*"funny_moments"[\s\S]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    result.funny_moments = parsed.get("funny_moments", [])

            # 更新帧信息
            for frame_data in frame_info:
                timestamp = frame_data['timestamp']
                frame_data["funny_score"] = self._calculate_frame_score(timestamp, result.funny_moments)
                frame_data["desc"] = self._get_frame_description(timestamp, result.funny_moments)

        except Exception as e:
            logger.error(f"解析响应失败: {e}")

        return result

    def _calculate_frame_score(self, timestamp: float, funny_moments: List[Dict]) -> float:
        """计算帧分数"""
        for moment in funny_moments:
            if moment["start_time"] <= timestamp <= moment["end_time"]:
                return moment.get("funny_score", 0.5)
        return 0.3

    def _get_frame_description(self, timestamp: float, funny_moments: List[Dict]) -> str:
        """获取帧描述"""
        for moment in funny_moments:
            if moment["start_time"] <= timestamp <= moment["end_time"]:
                return moment.get("description", "搞笑场景")
        return "普通场景"

    def _simulate_analysis(self, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """模拟分析"""
        result = VideoAnalysisResult()
        for i, frame_data in enumerate(frame_info):
            frame_data["funny_score"] = 0.8 + (i % 3) * 0.1
            frame_data["desc"] = f"模拟搞笑场景{i}"
        return result


class QwenVideoAnalyzer(BaseVideoAnalyzer):
    """通义千问视频分析器"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def test_connection(self) -> bool:
        """测试连接"""
        # TODO: 实现通义千问视频分析API测试
        logger.warning("通义千问视频分析暂未实现")
        return False

    def analyze_video(self, video_path: str, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """分析视频"""
        # TODO: 实现通义千问视频分析
        logger.warning("通义千问视频分析暂未实现,使用模拟模式")
        return self._simulate_analysis(frame_info, threshold)

    def _simulate_analysis(self, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """模拟分析"""
        result = VideoAnalysisResult()
        for i, frame_data in enumerate(frame_info):
            frame_data["funny_score"] = 0.8 + (i % 3) * 0.1
            frame_data["desc"] = f"模拟场景{i}"
        return result


class FrameByFrameAnalyzer(BaseVideoAnalyzer):
    """逐帧分析器(使用图像理解模型)"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            # 尝试调用analyze接口测试
            import io
            from PIL import Image

            # 创建测试图片
            test_image = Image.new('RGB', (100, 100), color='red')
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            test_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

            # 尝试方式1: /v1/analyze-frame接口
            data = {
                "image": test_base64,
                "prompt": "test"
            }

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            response = self.session.post(
                f"{self.base_url}/v1/analyze-frame",
                headers=headers,
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                logger.info("逐帧分析器连接成功(/v1/analyze-frame)")
                return True

            # 尝试方式2: /v1/chat/completions接口(兼容OpenAI)
            chat_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "评估这张图片的搞笑程度,只返回0-1的数字"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 10
            }

            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=chat_data,
                timeout=10
            )

            if response.status_code == 200:
                logger.info("逐帧分析器连接成功(/v1/chat/completions)")
                return True

            logger.warning(f"连接测试失败: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"连接测试异常: {e}")
            return False

    def analyze_video(self, video_path: str, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """逐帧分析视频"""
        result = VideoAnalysisResult()

        logger.info(f"开始逐帧分析视频,共{len(frame_info)}帧")

        for i, frame_data in enumerate(frame_info):
            try:
                frame_path = frame_data["path"]
                logger.info(f"分析第 {i+1}/{len(frame_info)} 帧: {frame_path}")

                # 调用图像理解API分析这一帧
                score, description = self._analyze_frame(frame_path)
                frame_data["funny_score"] = score
                frame_data["desc"] = description

                logger.debug(f"帧 {i+1} 搞笑分数: {score:.2f} - {description}")

            except Exception as e:
                logger.error(f"分析帧失败: {e}")
                frame_data["funny_score"] = 0.5
                frame_data["desc"] = "分析失败"

        return result

    def _analyze_frame(self, frame_path: str) -> tuple:
        """分析单帧,返回(score, description)"""
        try:
            # 读取图片并转base64
            with open(frame_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # 尝试方式1: 自定义/v1/analyze-frame接口
            score, desc = self._try_analyze_frame_api(image_data)
            if score is not None:
                return score, desc

            # 尝试方式2: OpenAI兼容/v1/chat/completions接口
            score, desc = self._try_chat_completions_api(image_data)
            if score is not None:
                return score, desc

            # 都失败,返回默认值
            return 0.5, "分析失败"

        except Exception as e:
            logger.error(f"帧分析失败: {e}")
            return 0.5, "分析异常"

    def _try_analyze_frame_api(self, image_base64: str) -> tuple:
        """尝试自定义analyze-frame接口"""
        try:
            data = {
                "image": image_base64,
                "prompt": "请评估这张图片的搞笑程度,返回0-1之间的分数和简短描述"
            }

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            response = self.session.post(
                f"{self.base_url}/v1/analyze-frame",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                score = float(result.get("funny_score", result.get("score", 0.5)))
                description = result.get("description", "场景分析")
                return score, description

        except Exception as e:
            logger.debug(f"analyze-frame接口失败: {e}")

        return None, None

    def _try_chat_completions_api(self, image_base64: str) -> tuple:
        """尝试OpenAI兼容chat/completions接口"""
        try:
            prompt = """请评估这张图片的搞笑程度,返回JSON格式:
{"funny_score": 0-1的分数, "description": "简短描述"}

评分标准:
- 0.9-1.0: 极度搞笑(夸张表情、意外场景)
- 0.7-0.9: 很搞笑(有趣动作、好笑情节)
- 0.5-0.7: 中等搞笑(轻微有趣元素)
- 0.3-0.5: 略有趣味
- 0-0.3: 普通场景"""

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.3
            }

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # 尝试解析JSON
                import json
                import re

                json_match = re.search(r'\{[\s\S]*"funny_score"[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    score = float(parsed.get("funny_score", 0.5))
                    description = parsed.get("description", "场景分析")
                    return score, description
                else:
                    # 如果没有JSON,尝试从文本提取分数
                    score_match = re.search(r'(\d+\.?\d*)', content)
                    if score_match:
                        score = float(score_match.group(1))
                        if score > 1:
                            score = score / 10  # 可能是10分制
                        return score, content[:50]

        except Exception as e:
            logger.debug(f"chat/completions接口失败: {e}")

        return None, None


class OllamaVideoAnalyzer(BaseVideoAnalyzer):
    """Ollama视频分析器(使用本地视觉模型逐帧分析)"""

    def __init__(self, base_url: str = 'http://localhost:11434', model: str = 'llava'):
        """
        初始化Ollama分析器

        Args:
            base_url: Ollama服务地址,默认 http://localhost:11434
            model: 视觉模型名称,如 llava, llava:13b, llava:34b, bakllava 等
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()

    def test_connection(self) -> bool:
        """测试Ollama连接"""
        try:
            # 1. 检查Ollama服务是否运行
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error(f"Ollama服务未响应: {response.status_code}")
                return False

            # 2. 检查模型是否已下载
            models_data = response.json()
            available_models = [m["name"] for m in models_data.get("models", [])]

            # 检查模型是否存在(支持带标签的模型名如llava:latest)
            model_exists = False
            for available in available_models:
                if available.startswith(self.model) or available.split(':')[0] == self.model:
                    model_exists = True
                    logger.info(f"找到Ollama模型: {available}")
                    break

            if not model_exists:
                logger.warning(f"Ollama模型 {self.model} 未找到,可用模型: {available_models}")
                logger.warning(f"请先运行: ollama pull {self.model}")
                return False

            # 3. 测试生成
            import io
            from PIL import Image

            test_image = Image.new('RGB', (100, 100), color='blue')
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            test_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

            data = {
                "model": self.model,
                "prompt": "What do you see?",
                "images": [test_base64],
                "stream": False
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Ollama连接测试成功(模型:{self.model})")
                return True
            else:
                logger.error(f"Ollama生成测试失败: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Ollama连接测试异常: {e}")
            return False

    def analyze_video(self, video_path: str, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """逐帧分析视频"""
        result = VideoAnalysisResult()

        logger.info(f"使用Ollama模型 {self.model} 逐帧分析视频,共{len(frame_info)}帧")

        for i, frame_data in enumerate(frame_info):
            try:
                frame_path = frame_data["path"]
                logger.info(f"分析第 {i+1}/{len(frame_info)} 帧: {frame_path}")

                # 调用Ollama视觉模型分析这一帧
                score, description = self._analyze_frame(frame_path)
                frame_data["funny_score"] = score
                frame_data["desc"] = description

                logger.debug(f"帧 {i+1} 搞笑分数: {score:.2f} - {description}")

            except Exception as e:
                logger.error(f"分析帧失败: {e}")
                frame_data["funny_score"] = 0.5
                frame_data["desc"] = "分析失败"

        return result

    def _analyze_frame(self, frame_path: str) -> tuple:
        """
        使用Ollama视觉模型分析单帧
        返回: (funny_score, description)
        """
        try:
            # 读取图片并转base64
            with open(frame_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            prompt = """请评估这张图片的搞笑程度,并以JSON格式返回结果:
{"funny_score": 0-1之间的分数, "description": "简短描述"}

评分标准:
- 0.9-1.0: 极度搞笑(夸张表情、意外场景、爆笑动作)
- 0.7-0.9: 很搞笑(有趣动作、好笑情节、滑稽表情)
- 0.5-0.7: 中等搞笑(轻微有趣元素)
- 0.3-0.5: 略有趣味
- 0-0.3: 普通场景

只返回JSON,不要额外解释。"""

            data = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "format": "json",  # 要求JSON输出
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")

                # 解析JSON响应
                import json
                import re

                # 尝试直接解析
                try:
                    parsed = json.loads(content)
                    score = float(parsed.get("funny_score", 0.5))
                    description = parsed.get("description", "场景分析")
                    return score, description
                except:
                    pass

                # 尝试提取JSON片段
                json_match = re.search(r'\{[\s\S]*"funny_score"[\s\S]*\}', content)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        score = float(parsed.get("funny_score", 0.5))
                        description = parsed.get("description", "场景分析")
                        return score, description
                    except:
                        pass

                # 如果JSON解析失败,尝试从文本提取分数
                score_match = re.search(r'(\d+\.?\d*)', content)
                if score_match:
                    score = float(score_match.group(1))
                    if score > 1:
                        score = score / 10  # 可能是10分制
                    description = content[:100] if content else "场景分析"
                    return min(score, 1.0), description

                # 都失败,返回默认值
                logger.warning(f"无法解析Ollama响应: {content}")
                return 0.5, content[:100] if content else "分析完成"

            else:
                logger.error(f"Ollama请求失败: {response.status_code}")
                return 0.5, "请求失败"

        except Exception as e:
            logger.error(f"Ollama帧分析异常: {e}")
            return 0.5, "分析异常"


class SimulateAnalyzer(BaseVideoAnalyzer):
    """模拟分析器"""

    def test_connection(self) -> bool:
        return True

    def analyze_video(self, video_path: str, frame_info: List[Dict], threshold: float) -> VideoAnalysisResult:
        """模拟分析"""
        result = VideoAnalysisResult()

        logger.info("使用模拟分析模式")

        for i, frame_data in enumerate(frame_info):
            funny_score = 0.8 + (i % 3) * 0.1
            frame_data["funny_score"] = funny_score
            frame_data["desc"] = f"模拟搞笑场景{i}"

        return result


# 支持的模式
SUPPORTED_ANALYZERS = {
    "gemini": {
        "name": "Google Gemini视频理解",
        "description": "使用Gemini 2.5 Flash分析整个视频"
    },
    "qwen_video": {
        "name": "通义千问视频分析",
        "description": "使用通义千问视频理解API"
    },
    "frame_by_frame": {
        "name": "逐帧分析(自定义API)",
        "description": "使用图像理解模型逐帧分析"
    },
    "ollama": {
        "name": "Ollama本地视觉模型",
        "description": "使用本地Ollama视觉模型(如llava)逐帧分析"
    },
    "simulate": {
        "name": "模拟模式",
        "description": "本地模拟,无需网络"
    }
}


if __name__ == "__main__":
    # 测试
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== 视频分析器测试 ===\n")

    # 测试模拟模式
    analyzer = UniversalVideoAnalyzer(mode="simulate")
    if analyzer.test_connection():
        print("✅ 模拟模式初始化成功")
    else:
        print("❌ 模拟模式初始化失败")

    print("\n支持的分析模式:")
    for key, info in SUPPORTED_ANALYZERS.items():
        print(f"  - {key}: {info['name']}")
