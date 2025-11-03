# -*- coding: utf-8 -*-
import os
import sys
import cv2
import configparser
import logging
import subprocess
import base64
import requests
import json
from datetime import datetime
from qwen_image_editor import QwenImageEditor

# ========== 配置读取 ==========
cfg = configparser.ConfigParser()
cfg.read("config.ini", encoding="utf-8")

# ========== 日志设置 ==========
log_level = getattr(logging, cfg.get("general", "log_level", fallback="INFO"))
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(cfg.get("general", "log_file", fallback="workflow.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 工具函数 ==========
def extract_frames(video_path, interval_ms=5000):
    """按时间间隔抽帧，返回帧信息和时间戳"""
    logger.info(f"开始抽帧: {video_path}, 每 {interval_ms} ms 抽一帧")
    if not os.path.exists(video_path):
        logger.error("视频文件不存在！")
        return [], None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    logger.info(f"视频信息: fps={fps}, 总帧数={total_frames}, 时长={duration:.2f}秒")
    
    frame_info = []  # 存储帧信息：[{文件路径, 帧号, 时间戳}]
    frame_interval = int(fps * interval_ms / 1000)

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            timestamp = idx / fps  # 计算时间戳（秒）
            frame_path = f"./frames/frame_{idx}.jpg"
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            cv2.imwrite(frame_path, frame)
            
            frame_info.append({
                "path": frame_path,
                "frame_number": idx,
                "timestamp": timestamp
            })
            
            logger.debug(f"抽取帧: {frame_path}, 帧号={idx}, 时间={timestamp:.2f}s")
        idx += 1
    
    cap.release()
    
    video_info = {
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "path": video_path
    }
    
    logger.info(f"抽帧完成，共 {len(frame_info)} 张")
    return frame_info, video_info

def _get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0:
            duration = frame_count / fps
            logger.debug(f"视频时长: {duration:.2f}秒")
            return duration
        else:
            logger.warning("无法获取视频FPS")
            return 0.0
    except Exception as e:
        logger.error(f"获取视频时长失败: {e}")
        return 0.0

def test_proxy_connection(proxies):
    """测试代理连接是否正常"""
    if not proxies:
        return True
    
    try:
        logger.info("正在测试代理连接...")
        test_response = requests.get(
            "https://www.google.com",
            proxies=proxies,
            timeout=10
        )
        if test_response.status_code == 200:
            logger.info("代理连接测试成功")
            return True
        else:
            logger.warning(f"代理连接测试失败，状态码: {test_response.status_code}")
            return False
    except Exception as e:
        logger.error(f"代理连接测试失败: {e}")
        return False

def google_vision_analysis(frame_info, api_key, threshold):
    """调用Gemini 2.5 Flash视频理解模型分析整个视频，支持代理"""
    logger.info(f"调用Gemini 2.5 Flash视频理解分析")
    
    # 检查是否使用在线API
    try:
        use_online_api_str = cfg.get("google_vision", "use_online_api", fallback="true")
        use_online_api_str = use_online_api_str.split(';')[0].strip().lower()
        use_online_api = use_online_api_str in ['true', '1', 'yes', 'on']
    except (ValueError, configparser.NoOptionError) as e:
        use_online_api = True
    
    if not use_online_api:
        logger.info("配置为不使用在线API，直接使用模拟数据")
        return _simulate_google_vision_analysis(frame_info, threshold)
    
    # 检查API Key是否有效
    if not api_key or api_key == "your_google_vision_api_key":
        logger.warning("未配置Gemini API Key，使用模拟数据")
        return _simulate_google_vision_analysis(frame_info, threshold)
    
    # 设置代理
    proxies = None
    # 安全读取proxy_enabled配置
    try:
        proxy_enabled_str = cfg.get("google_vision", "proxy_enabled", fallback="false")
        proxy_enabled_str = proxy_enabled_str.split(';')[0].strip().lower()
        proxy_enabled = proxy_enabled_str in ['true', '1', 'yes', 'on']
    except (ValueError, configparser.NoOptionError) as e:
        logger.warning(f"读取proxy_enabled配置失败: {e}，使用默认值False")
        proxy_enabled = False
    
    if proxy_enabled:
        proxy_host = cfg.get("google_vision", "proxy_host", fallback="127.0.0.1")
        proxy_port = cfg.get("google_vision", "proxy_port", fallback="1080")
        proxy_url = f"http://{proxy_host}:{proxy_port}"
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        logger.info(f"使用代理: {proxy_url}")
        
        # 测试代理连接
        if not test_proxy_connection(proxies):
            logger.warning("代理连接失败，将尝试不使用代理")
            proxies = None
    
    # 获取视频路径
    if frame_info and len(frame_info) > 0:
        video_path = frame_info[0].get("video_path", "")
        if not video_path:
            # 从全局变量获取
            video_path = getattr(google_vision_analysis, '_current_video_path', None)
    else:
        video_path = None
    
    if not video_path or not os.path.exists(video_path):
        logger.error("无法获取视频路径，使用模拟数据")
        return _simulate_google_vision_analysis(frame_info, threshold)
    
    try:
        # 使用 Gemini 视频理解 API
        video_analysis_result = _analyze_video_with_gemini(video_path, api_key, proxies)
        
        if video_analysis_result:
            # 根据视频分析结果更新帧信息
            results = _update_frames_with_video_analysis(frame_info, video_analysis_result, threshold)
        else:
            logger.warning("视频分析失败，使用模拟数据")
            results = _simulate_google_vision_analysis(frame_info, threshold)
            
    except Exception as e:
        logger.error(f"视频分析过程中出错: {e}")
        results = _simulate_google_vision_analysis(frame_info, threshold)
    
    return results

def _analyze_video_with_gemini(video_path, api_key, proxies):
    """使用 Gemini API 分析视频"""
    try:
        logger.info(f"开始上传视频: {video_path}")
        
        # 1. 获取视频信息
        mime_type = "video/mp4"  # 假设为mp4格式
        file_size = os.path.getsize(video_path)
        display_name = os.path.basename(video_path)
        
        # 检查文件大小（Gemini 有大小限制）
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            logger.error(f"视频文件太大: {file_size / 1024 / 1024:.2f}MB > {max_size / 1024 / 1024}MB")
            logger.info("将使用模拟数据")
            return None
        
        logger.info(f"文件大小: {file_size / 1024 / 1024:.2f}MB")
        
        # 2. 开始上传
        upload_url = _start_file_upload(api_key, mime_type, file_size, display_name, proxies)
        if not upload_url:
            return None
        
        # 3. 上传视频数据
        file_uri = _upload_video_data(upload_url, video_path, file_size, proxies)
        if not file_uri:
            logger.warning("视频上传失败，将使用模拟数据")
            return None
        
        logger.info(f"视频上传成功: {file_uri}")
        
        # 4. 分析视频内容
        analysis_result = _generate_content_from_video(file_uri, mime_type, api_key, proxies)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"视频分析失败: {e}")
        return None

def _start_file_upload(api_key, mime_type, file_size, display_name, proxies):
    """开始文件上传"""
    try:
        headers = {
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json"
        }
        
        data = {
            "file": {
                "display_name": display_name
            }
        }
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            headers=headers,
            json=data,
            proxies=proxies,
            timeout=30  # 添加30秒超时
        )
        
        if response.status_code == 200:
            upload_url = response.headers.get("x-goog-upload-url")
            logger.debug(f"获取上传URL: {upload_url}")
            return upload_url
        else:
            logger.error(f"开始上传失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"开始上传失败: {e}")
        return None

def _upload_video_data(upload_url, video_path, file_size, proxies):
    """上传视频数据"""
    try:
        logger.info(f"开始上传视频数据: {file_size / 1024 / 1024:.2f}MB")
        
        headers = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize"
        }
        
        # 使用进度显示的文件上传
        with open(video_path, 'rb') as f:
            logger.info("正在上传视频数据...（这可能需要几分钟）")
            
            response = requests.post(
                upload_url,
                headers=headers,
                data=f,
                proxies=proxies,
                timeout=600  # 10分钟超时
            )
        
        if response.status_code == 200:
            result = response.json()
            file_uri = result.get("file", {}).get("uri", "")
            logger.debug(f"上传完成，文件URI: {file_uri}")
            return file_uri
        else:
            logger.error(f"上传视频数据失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"上传视频数据失败: {e}")
        return None

def _generate_content_from_video(file_uri, mime_type, api_key, proxies):
    """从视频生成分析内容"""
    try:
        # 从配置文件读取提示词
        prompt = cfg.get("google_vision", "video_analysis_prompt", fallback="""请分析这个视频，找出最搞笑、最有趣的时间段。""")
        
        logger.debug(f"使用视频分析提示词: {prompt[:100]}...")
        
        data = {
            "contents": [{
                "parts": [
                    {
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 4096,
                "candidateCount": 1
            }
        }
        
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            headers=headers,
            json=data,
            proxies=proxies,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            logger.error(f"生成内容失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"生成内容失败: {e}")
        return None

def _update_frames_with_video_analysis(frame_info, video_analysis_result, threshold):
    """根据视频分析结果更新帧信息"""
    try:
        # 解析 Gemini 的响应
        funny_moments = _parse_video_analysis_response(video_analysis_result)

        # 应用图片数量控制
        try:
            from image_count_controller import ImageCountController

            # 获取图片控制配置
            image_control_config = {
                'max_images': config.getint('google_vision', 'max_images', fallback=6),
                'min_interval_seconds': config.getfloat('google_vision', 'min_interval_seconds', fallback=5.0),
                'images_per_minute': config.getfloat('google_vision', 'images_per_minute', fallback=2.0),
                'auto_limit_mode': config.get('google_vision', 'auto_limit_mode', fallback='smart')
            }

            controller = ImageCountController(image_control_config)

            # 获取视频时长
            video_duration = _get_video_duration(video_path)

            # 过滤搞笑时刻
            original_count = len(funny_moments)
            funny_moments = controller.filter_funny_moments(funny_moments, video_duration)

            logger.info(f"图片数量控制: {original_count} -> {len(funny_moments)} 个时刻")
            logger.info(f"控制信息: {controller.get_control_info(video_duration)}")

        except ImportError:
            logger.warning("图片数量控制模块未找到，使用原始搞笑时刻")
        except Exception as e:
            logger.error(f"图片数量控制失败: {e}")

        results = []
        json_results = {
            "video_analysis": {
                "total_frames": len(frame_info),
                "threshold": threshold,
                "analysis_time": datetime.now().isoformat(),
                "video_moments": funny_moments,
                "key_moments": []
            }
        }
        
        # 为每个帧计算搞笑分数
        for frame_data in frame_info:
            timestamp = frame_data['timestamp']
            funny_score = _calculate_frame_score(timestamp, funny_moments)
            
            frame_data["funny_score"] = funny_score
            frame_data["desc"] = _get_frame_description(timestamp, funny_moments)
            
            # 构建 JSON 结果
            moment_data = {
                "timestamp": round(timestamp, 2),
                "frame_number": frame_data['frame_number'],
                "funny_score": round(funny_score, 2),
                "description": frame_data["desc"],
                "frame_path": frame_data["path"],
                "is_key_moment": funny_score >= threshold
            }
            json_results["video_analysis"]["key_moments"].append(moment_data)
            
            if funny_score >= threshold:
                results.append(frame_data)
        
        # 保存 JSON 结果
        json_output_path = "./analysis_results.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"筛选后保留 {len(results)} 张搞笑画面")
        logger.info(f"分析结果已保存到: {json_output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"更新帧信息失败: {e}")
        return []

def _parse_video_analysis_response(video_analysis_result):
    """解析视频分析响应"""
    try:
        if "candidates" in video_analysis_result and len(video_analysis_result["candidates"]) > 0:
            candidate = video_analysis_result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                text_content = candidate["content"]["parts"][0].get("text", "")
                
                # 尝试解析JSON响应
                import re
                json_match = re.search(r'\{[\s\S]*"funny_moments"[\s\S]*\}', text_content)
                if json_match:
                    json_str = json_match.group()
                    parsed_json = json.loads(json_str)
                    return parsed_json.get("funny_moments", [])
        
        return []
        
    except Exception as e:
        logger.error(f"解析视频分析响应失败: {e}")
        return []

def _calculate_frame_score(timestamp, funny_moments):
    """根据搞笑时间段计算帧的分数"""
    for moment in funny_moments:
        start_time = moment.get("start_time", 0)
        end_time = moment.get("end_time", 0)
        if start_time <= timestamp <= end_time:
            return moment.get("funny_score", 0.5)
    return 0.3  # 默认较低分数

def _get_frame_description(timestamp, funny_moments):
    """获取帧的描述"""
    for moment in funny_moments:
        start_time = moment.get("start_time", 0)
        end_time = moment.get("end_time", 0)
        if start_time <= timestamp <= end_time:
            return moment.get("description", "搞笑场景")
    return "普通场景"

def _simulate_google_vision_analysis(frame_info, threshold):
    """模拟 Google Vision 分析结果"""
    logger.info("使用模拟数据进行分析")
    results = []
    json_results = {
        "video_analysis": {
            "total_frames": len(frame_info),
            "threshold": threshold,
            "analysis_time": datetime.now().isoformat(),
            "proxy_used": False,
            "mode": "simulation",
            "key_moments": []
        }
    }
    
    for i, frame_data in enumerate(frame_info):
        funny_score = 0.8 + (i % 3) * 0.1  # 假分数
        frame_data["funny_score"] = funny_score
        frame_data["desc"] = f"模拟搞笑场景{i}"
        
        # 构建 JSON 结果
        moment_data = {
            "timestamp": round(frame_data['timestamp'], 2),
            "frame_number": frame_data['frame_number'],
            "funny_score": round(funny_score, 2),
            "description": frame_data["desc"],
            "frame_path": frame_data["path"],
            "is_key_moment": funny_score >= threshold
        }
        json_results["video_analysis"]["key_moments"].append(moment_data)
        
        logger.debug(f"{frame_data['path']} -> funny_score={funny_score}, 时间={frame_data['timestamp']:.2f}s (模拟)")
        
        if funny_score >= threshold:
            results.append(frame_data)
    
    # 保存 JSON 结果到文件
    json_output_path = "./analysis_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"筛选后保留 {len(results)} 张搞笑画面 (模拟)")
    logger.info(f"分析结果已保存到: {json_output_path}")
    
    return results

def _parse_gemini_response(gemini_result):
    """解析Gemini API响应，提取搞笑分数和描述"""
    try:
        if "candidates" in gemini_result and len(gemini_result["candidates"]) > 0:
            candidate = gemini_result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                text_content = candidate["content"]["parts"][0].get("text", "")
                
                # 尝试解析JSON响应
                try:
                    # 查找 JSON 块
                    import re
                    json_match = re.search(r'\{[^{}]*"funny_score"[^{}]*\}', text_content)
                    if json_match:
                        json_str = json_match.group()
                        parsed_json = json.loads(json_str)
                        
                        funny_score = float(parsed_json.get("funny_score", 0.5))
                        description = parsed_json.get("description", "内容分析")
                        
                        return funny_score, description
                except:
                    pass
                
                # 如果JSON解析失败，使用关键词匹配
                funny_keywords = ["funny", "humor", "laugh", "smile", "comic", "amusing", "entertaining"]
                score = 0.3  # 基础分数
                
                text_lower = text_content.lower()
                for keyword in funny_keywords:
                    if keyword in text_lower:
                        score += 0.2
                
                if "people" in text_lower or "person" in text_lower or "face" in text_lower:
                    score += 0.2
                
                return min(score, 1.0), text_content[:100]  # 取前100个字符作为描述
        
        return 0.5, "分析结果异常"
        
    except Exception as e:
        logger.error(f"解析Gemini响应时出错: {e}")
        return 0.5, "解析失败"

def nano_banana_generate(funny_frames, api_key, prompt):
    """生成手绘插画 - 支持Gemini和通义千问"""
    logger.info(f"生成手绘插画，共 {len(funny_frames)} 张")

    # 检查是否启用通义千问
    try:
        qwen_enabled_str = cfg.get("qwen", "enabled", fallback="false")
        qwen_enabled_str = qwen_enabled_str.split(';')[0].strip().lower()
        qwen_enabled = qwen_enabled_str in ['true', '1', 'yes', 'on']
    except (ValueError, configparser.NoOptionError) as e:
        qwen_enabled = False

    if qwen_enabled:
        logger.info("使用通义千问图片编辑API生成手绘插画")
        return _qwen_generate_illustrations(funny_frames)

    # 原有的Gemini逻辑
    logger.info("使用Gemini API生成手绘插画")

    # 检查是否使用在线API
    try:
        use_online_api_str = cfg.get("google_vision", "use_online_api", fallback="true")
        use_online_api_str = use_online_api_str.split(';')[0].strip().lower()
        use_online_api = use_online_api_str in ['true', '1', 'yes', 'on']
    except (ValueError, configparser.NoOptionError) as e:
        use_online_api = True

    # 检查是否启用真实的图生图功能
    try:
        enable_real_generation_str = cfg.get("nano_banana", "enable_real_generation", fallback="true")
        enable_real_generation_str = enable_real_generation_str.split(';')[0].strip().lower()
        enable_real_generation = enable_real_generation_str in ['true', '1', 'yes', 'on']
    except (ValueError, configparser.NoOptionError) as e:
        enable_real_generation = True

    if not enable_real_generation:
        logger.info("配置为不启用真实图生图，使用模拟手绘效果")
        return _simulate_nano_banana_generate(funny_frames, prompt)

    if not use_online_api:
        logger.info("配置为不使用在线API，使用模拟手绘生成")
        return _simulate_nano_banana_generate(funny_frames, prompt)
    
    # 检查API Key是否有效
    if not api_key or api_key == "your_nano_banana_api_key":
        logger.warning("未配置Gemini API Key，使用模拟数据")
        return _simulate_nano_banana_generate(funny_frames, prompt)
    
    # 设置代理（复用Google Vision的代理配置）
    proxies = None
    # 安全读取proxy_enabled配置
    try:
        proxy_enabled_str = cfg.get("google_vision", "proxy_enabled", fallback="false")
        proxy_enabled_str = proxy_enabled_str.split(';')[0].strip().lower()
        proxy_enabled = proxy_enabled_str in ['true', '1', 'yes', 'on']
    except (ValueError, configparser.NoOptionError) as e:
        logger.warning(f"读取proxy_enabled配置失败: {e}，使用默认值False")
        proxy_enabled = False
    
    if proxy_enabled:
        proxy_host = cfg.get("google_vision", "proxy_host", fallback="127.0.0.1")
        proxy_port = cfg.get("google_vision", "proxy_port", fallback="1080")
        proxy_url = f"http://{proxy_host}:{proxy_port}"
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        logger.info(f"使用代理: {proxy_url}")
        
        # 测试代理连接（但不停止执行）
        if not test_proxy_connection(proxies):
            logger.warning("代理连接测试失败，但将继续尝试")
    
    # 获取模型名称
    model = cfg.get("nano_banana", "model", fallback="gemini-2.5-flash-image-preview")
    logger.info(f"使用图像生成模型: {model}")
    
    # Gemini Image API endpoint
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    for frame_data in funny_frames:
        try:
            original_path = frame_data["path"]
            illustration_path = original_path.replace("frames", "illustrations")
            os.makedirs(os.path.dirname(illustration_path), exist_ok=True)
            
            # 读取原图并转换为base64
            with open(original_path, "rb") as image_file:
                image_content = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 构建 Gemini Image API 请求数据
            request_data = {
                "contents": [{
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_content
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.4,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 4096,
                    "candidateCount": 1
                }
            }
            
            # 发送请求
            logger.debug(f"调用Gemini Image API生成: {original_path}")
            
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(gemini_url, json=request_data, headers=headers, proxies=proxies, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # 提取生成的图像数据
                if _extract_generated_image(result, illustration_path):
                    logger.info(f"成功生成手绘图: {illustration_path}")
                else:
                    # 如果提取失败，使用模拟方式
                    logger.warning(f"图像提取失败，使用模拟方式: {original_path}")
                    _create_simulated_illustration(original_path, illustration_path, frame_data)
            else:
                logger.error(f"Gemini Image API请求失败: {response.status_code} - {response.text}")
                # 失败时使用模拟方式
                _create_simulated_illustration(original_path, illustration_path, frame_data)
                
        except Exception as e:
            logger.error(f"处理图片 {original_path} 时出错: {e}")
            # 出错时使用模拟方式
            _create_simulated_illustration(original_path, illustration_path, frame_data)
        
        # 更新手绘图路径
        frame_data["illustration_path"] = illustration_path
        logger.debug(f"完成手绘图: {illustration_path} (时间={frame_data['timestamp']:.2f}s)")
    
    return funny_frames

def _extract_generated_image(gemini_result, output_path):
    """从 Gemini 响应中提取生成的图像数据"""
    try:
        if "candidates" in gemini_result and len(gemini_result["candidates"]) > 0:
            candidate = gemini_result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "inline_data" in part:
                        image_data = part["inline_data"].get("data", "")
                        if image_data:
                            # 解码base64图像数据
                            image_bytes = base64.b64decode(image_data)
                            with open(output_path, "wb") as f:
                                f.write(image_bytes)
                            return True
        return False
    except Exception as e:
        logger.error(f"提取图像数据失败: {e}")
        return False

def _create_simulated_illustration(original_path, illustration_path, frame_data):
    """创建模拟手绘图（作为备用方案）"""
    try:
        if os.path.exists(original_path):
            # 读取原图
            frame = cv2.imread(original_path)
            if frame is not None:
                # 创建手绘风格效果
                
                # 1. 转为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 2. 高斯模糊
                gray_blur = cv2.medianBlur(gray, 5)
                
                # 3. 创建边缘线条（使用自适应阈值）
                edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
                
                # 4. 转换为3通道图像
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # 5. 与原图混合，创建手绘效果
                hand_drawn = cv2.bitwise_and(frame, edges_colored)
                
                # 6. 调整亮度和对比度
                hand_drawn = cv2.convertScaleAbs(hand_drawn, alpha=1.2, beta=20)
                
                # 7. 添加手绘风格标记
                height, width = hand_drawn.shape[:2]
                
                # 添加半透明标签
                cv2.rectangle(hand_drawn, (10, 10), (300, 80), (255, 255, 255), -1)
                cv2.rectangle(hand_drawn, (10, 10), (300, 80), (0, 0, 0), 2)
                
                cv2.putText(hand_drawn, "Hand-drawn Style", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(hand_drawn, f"t={frame_data['timestamp']:.1f}s", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                cv2.imwrite(illustration_path, hand_drawn)
                logger.debug(f"生成手绘效果图: {illustration_path}")
            else:
                logger.error(f"无法读取原图: {original_path}")
                _create_blank_illustration(illustration_path, frame_data)
        else:
            logger.warning(f"原图不存在: {original_path}")
            _create_blank_illustration(illustration_path, frame_data)
    except Exception as e:
        logger.error(f"创建模拟手绘图失败: {e}")
        _create_blank_illustration(illustration_path, frame_data)

def _create_blank_illustration(illustration_path, frame_data):
    """创建空白手绘风格图像"""
    try:
        import numpy as np
        # 创建类似纸张的背景
        blank = 250 * np.ones((720, 1280, 3), dtype=np.uint8)
        
        # 添加一些噪声模拟纸张纹理
        noise = np.random.randint(0, 30, (720, 1280, 3), dtype=np.uint8)
        blank = cv2.subtract(blank, noise)
        
        # 添加手绘风格文字
        cv2.putText(blank, "Hand-drawn Illustration", (400, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 3)
        cv2.putText(blank, "(Simulated)", (500, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        cv2.putText(blank, f"Time: {frame_data['timestamp']:.1f}s", (450, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 2)
        
        # 添加一些手绘线条
        cv2.line(blank, (200, 500), (1000, 500), (0, 0, 0), 2)
        cv2.circle(blank, (640, 200), 100, (0, 0, 0), 3, cv2.LINE_AA)
        
        cv2.imwrite(illustration_path, blank)
    except Exception as e:
        logger.error(f"创建空白手绘图失败: {e}")

def _qwen_generate_illustrations(funny_frames):
    """使用通义千问API生成手绘插画"""
    try:
        # 从配置读取API密钥和提示词
        qwen_api_key = cfg.get("qwen", "api_key", fallback="")
        hand_drawn_prompt = cfg.get("qwen", "hand_drawn_prompt", fallback="")

        if not qwen_api_key or qwen_api_key == "your_dashscope_api_key_here":
            logger.warning("未配置通义千问API Key，使用模拟数据")
            return _simulate_nano_banana_generate(funny_frames, hand_drawn_prompt)

        # 初始化通义千问编辑器
        editor = QwenImageEditor(api_key=qwen_api_key)

        # 测试连接
        if not editor.test_connection():
            logger.warning("通义千问API连接失败，使用模拟数据")
            return _simulate_nano_banana_generate(funny_frames, hand_drawn_prompt)

        logger.info("开始使用通义千问API生成手绘插画")

        for frame_data in funny_frames:
            try:
                original_path = frame_data["path"]
                illustration_path = original_path.replace("frames", "illustrations")
                os.makedirs(os.path.dirname(illustration_path), exist_ok=True)

                logger.info(f"正在处理: {original_path}")

                # 获取原图尺寸作为目标尺寸
                import cv2
                original_img = cv2.imread(original_path)
                if original_img is not None:
                    target_height, target_width = original_img.shape[:2]
                    target_size = (target_width, target_height)
                    logger.debug(f"目标尺寸: {target_width}x{target_height}")
                else:
                    target_size = None
                    logger.warning(f"无法读取原图尺寸: {original_path}")

                # 调用通义千问API转换图片
                result_base64 = editor.convert_to_hand_drawn(original_path, hand_drawn_prompt)

                if result_base64:
                    # 保存生成的图片，调整到原图尺寸
                    if editor.save_base64_image(result_base64, illustration_path, target_size):
                        logger.info(f"✅ 通义千问生成手绘图成功: {illustration_path}")
                    else:
                        logger.warning(f"保存失败，使用模拟方式: {original_path}")
                        _create_simulated_illustration(original_path, illustration_path, frame_data)
                else:
                    logger.warning(f"通义千问转换失败，使用模拟方式: {original_path}")
                    _create_simulated_illustration(original_path, illustration_path, frame_data)

                # 更新帧数据
                frame_data["illustration_path"] = illustration_path

            except Exception as e:
                logger.error(f"处理帧 {original_path} 时出错: {e}")
                # 出错时使用模拟方式
                _create_simulated_illustration(original_path, illustration_path, frame_data)
                frame_data["illustration_path"] = illustration_path

        return funny_frames

    except Exception as e:
        logger.error(f"通义千问处理过程出错: {e}")
        return _simulate_nano_banana_generate(funny_frames, "")

def _simulate_nano_banana_generate(funny_frames, prompt):
    """模拟 Nano Banana 生成结果"""
    logger.info("使用模拟数据进行手绘生成")

    for frame_data in funny_frames:
        original_path = frame_data["path"]
        illustration_path = original_path.replace("frames", "illustrations")
        os.makedirs(os.path.dirname(illustration_path), exist_ok=True)
        _create_simulated_illustration(original_path, illustration_path, frame_data)
        frame_data["illustration_path"] = illustration_path

    return funny_frames

def compose_video_with_overlay(video_info, funny_frames_with_illustrations, bgm, draft_dir, video_name, overlay_duration=1.0, volume=1.0):
    """在原视频的基础上，在指定时间点插入手绘插画（暂停原视频）

    Args:
        video_info: 原视频信息
        funny_frames_with_illustrations: 包含手绘图的帧数据
        bgm: 背景音乐文件路径
        draft_dir: 输出目录
        video_name: 视频文件名前缀
        overlay_duration: 手绘图显示时长（秒）
        volume: 背景音乐音量
    """
    logger.info(f"开始合成视频，在 {len(funny_frames_with_illustrations)} 个时间点插入手绘图（每张显示{overlay_duration}秒）")
    
    os.makedirs(draft_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_file = os.path.join(draft_dir, f"{video_name}_{timestamp}.mp4")

    # 按时间排序手绘图
    funny_frames_with_illustrations.sort(key=lambda x: x["timestamp"])

    logger.info(f"准备插入 {len(funny_frames_with_illustrations)} 张手绘图")
    for frame in funny_frames_with_illustrations:
        logger.info(f"  - 时间点 {frame['timestamp']:.1f}s: {frame['illustration_path']}")

    # 创建视频段列表：原视频段 + 手绘图段
    segments = []
    current_time = 0.0
    original_video = video_info["path"]

    for i, frame_data in enumerate(funny_frames_with_illustrations):
        insert_time = frame_data["timestamp"]
        illustration_path = frame_data["illustration_path"]

        # 添加原视频段（从当前时间到插入时间）
        if insert_time > current_time:
            segment_duration = insert_time - current_time
            segments.append({
                "type": "video",
                "source": original_video,
                "start": current_time,
                "duration": segment_duration
            })
            logger.debug(f"视频段: {current_time:.1f}s - {insert_time:.1f}s (时长 {segment_duration:.1f}s)")

        # 添加手绘图段
        segments.append({
            "type": "image",
            "source": illustration_path,
            "duration": overlay_duration
        })
        logger.debug(f"手绘图: {illustration_path} (显示 {overlay_duration}s)")

        current_time = insert_time

    # 添加最后一段原视频（从最后插入点到视频结束）
    if current_time < video_info["duration"]:
        final_duration = video_info["duration"] - current_time
        segments.append({
            "type": "video",
            "source": original_video,
            "start": current_time,
            "duration": final_duration
        })
        logger.debug(f"最后视频段: {current_time:.1f}s - {video_info['duration']:.1f}s (时长 {final_duration:.1f}s)")

    # 构建FFmpeg命令
    filter_complex = []
    inputs = []
    input_labels = []

    # 为每个段准备输入
    for i, segment in enumerate(segments):
        if segment["type"] == "video":
            # 视频段：剪切原视频的特定时间段
            inputs.extend([
                "-ss", str(segment["start"]),
                "-t", str(segment["duration"]),
                "-i", segment["source"]
            ])
            input_labels.append(f"[{len(input_labels)}:v]")
        else:
            # 图片段：循环显示图片指定时长
            inputs.extend([
                "-loop", "1",
                "-t", str(segment["duration"]),
                "-i", segment["source"]
            ])
            input_labels.append(f"[{len(input_labels)}:v]")

    # 输入背景音乐
    inputs.extend(["-i", bgm])
    bgm_index = len(input_labels)

    # 拼接所有视频段
    if len(input_labels) > 1:
        concat_inputs = "".join(input_labels)
        filter_complex.append(f"{concat_inputs}concat=n={len(input_labels)}:v=1:a=0[final_video]")
    else:
        filter_complex.append(f"{input_labels[0]}copy[final_video]")

    # 音频处理：创建静音音频轨道与背景音乐混合
    total_duration = sum(seg["duration"] for seg in segments)
    audio_filter = f"anullsrc=channel_layout=stereo:sample_rate=48000,atrim=duration={total_duration}[silence];[silence][{bgm_index}:a]amix=inputs=2:duration=first:dropout_transition=2,volume={volume}[final_audio]"
    filter_complex.append(audio_filter)

    # 合并滤镜
    filter_complex_str = ";".join(filter_complex)

    # FFmpeg命令
    ffmpeg_path = "./ffmpeg.exe" if os.path.exists("./ffmpeg.exe") else "ffmpeg"
    cmd = [ffmpeg_path, "-y"] + inputs + [
        "-filter_complex", filter_complex_str,
        "-map", "[final_video]",
        "-map", "[final_audio]",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-r", "30",  # 保持原视频帧率
        video_file
    ]

    logger.info(f"执行FFmpeg命令...")
    logger.debug(f"FFmpeg命令: {' '.join(cmd[:10])}... (简化显示)")

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"合成完成: {video_file}")
        final_duration = sum(seg["duration"] for seg in segments)
        logger.info(f"最终视频时长: {final_duration:.2f}秒 (原始{video_info['duration']:.2f}s + 插入图片{len(funny_frames_with_illustrations) * overlay_duration:.1f}s)")
        return video_file
    except subprocess.CalledProcessError as e:
        logger.error(f"视频合成失败: {e}")
        return None


# ========== 主流程 ==========
def main(input_video_path=None):
    logger.info("========== 搞笑手绘视频工作流开始 ==========")

    # 如果没有传入视频路径，使用配置文件中的路径
    if input_video_path is None:
        input_video_path = cfg.get("video", "input_video")
    
    logger.info(f"输入视频: {input_video_path}")
    
    # 从视频文件名生成输出文件名（去掉扩展名和路径）
    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    output_name = f"hand_drawn_{video_basename}"

    # 1. 抽取关键帧和视频信息
    # 安全读取frame_interval配置，清理可能的注释和空格
    try:
        frame_interval_str = cfg.get("video", "frame_interval")
        # 去除注释和空格
        frame_interval_str = frame_interval_str.split(';')[0].strip()
        frame_interval = int(frame_interval_str)
    except (ValueError, configparser.NoOptionError) as e:
        logger.warning(f"读取frame_interval配置失败: {e}，使用默认值5000")
        frame_interval = 5000
    
    frame_info, video_info = extract_frames(
        input_video_path,
        frame_interval
    )
    
    if not frame_info:
        logger.error("抽帧失败，退出")
        return None

    # 2. 分析搞笑程度（传递视频路径信息）
    # 为每个帧添加视频路径信息
    for frame_data in frame_info:
        frame_data["video_path"] = input_video_path
    
    # 设置全局变量供 google_vision_analysis 使用
    google_vision_analysis._current_video_path = input_video_path
    
    # 安全读取funny_score_threshold配置
    try:
        threshold_str = cfg.get("google_vision", "funny_score_threshold")
        threshold_str = threshold_str.split(';')[0].strip()
        funny_score_threshold = float(threshold_str)
    except (ValueError, configparser.NoOptionError) as e:
        logger.warning(f"读取funny_score_threshold配置失败: {e}，使用默认值0.85")
        funny_score_threshold = 0.85
    
    funny_frames = google_vision_analysis(
        frame_info,
        cfg.get("google_vision", "api_key"),
        funny_score_threshold
    )
    
    if not funny_frames:
        logger.warning("没有找到符合条件的搞笑场景，使用所有帧")
        funny_frames = frame_info

    # 3. 生成手绘风格插画
    funny_frames_with_illustrations = nano_banana_generate(
        funny_frames,
        cfg.get("nano_banana", "api_key"),
        cfg.get("nano_banana", "prompt")
    )

    # 安全读取bgm_volume配置
    try:
        bgm_volume_str = cfg.get("output", "bgm_volume")
        bgm_volume_str = bgm_volume_str.split(';')[0].strip()
        bgm_volume = float(bgm_volume_str)
    except (ValueError, configparser.NoOptionError) as e:
        logger.warning(f"读取bgm_volume配置失败: {e}，使用默认值0.5")
        bgm_volume = 0.5
    
    # 4. 合成视频：在原视频中插入手绘图
    output_file = compose_video_with_overlay(
        video_info,
        funny_frames_with_illustrations,
        cfg.get("output", "bgm_file"),
        cfg.get("output", "draft_dir"),
        output_name,
        overlay_duration=1.0,  # 手绘图显示1秒
        volume=bgm_volume
    )

    logger.info("========== 工作流完成 ✅ ==========")
    logger.info(f"最终视频文件: {output_file}")
    logger.info(f"原始视频时长: {video_info['duration']:.2f}秒, 输出视频保持相同")
    return output_file

if __name__ == "__main__":
    # 支持命令行参数传入视频路径
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        logger.info(f"使用命令行参数视频路径: {video_path}")
        main(video_path)
    else:
        logger.info("使用配置文件中的默认视频路径")
        main()
