# Ollama 视频分析使用指南

## 概述

现在你的商业软件"灵魂画手"已经完全支持 **Ollama 本地视觉模型**进行视频分析!

这意味着客户可以:
- ✅ 完全本地运行,无需网络
- ✅ 数据隐私保护,视频不上传云端
- ✅ 免费使用,无API费用
- ✅ 支持多种视觉模型(llava, llava:13b, bakllava等)

## 安装 Ollama

### Windows
1. 访问 https://ollama.com/download
2. 下载并安装 Windows 版本
3. 安装后 Ollama 会自动运行在后台(默认端口11434)

### 下载视觉模型

Ollama 支持多个视觉理解模型:

```bash
# 推荐模型(速度快,效果好)
ollama pull llava

# 更大模型(效果更好,速度较慢)
ollama pull llava:13b
ollama pull llava:34b

# 其他视觉模型
ollama pull bakllava
ollama pull llava-phi3
ollama pull llava-llama3
```

### 验证安装

```bash
# 查看已安装的模型
ollama list

# 测试视觉模型
ollama run llava "Describe this image" --image test.jpg
```

## 配置软件使用 Ollama

### 方法1: 修改 config.ini

在 `config.ini` 中添加或修改 `[video_analysis]` 配置段:

```ini
[video_analysis]
; 设置分析模式为 ollama
mode = ollama
enabled = true

; Ollama 服务地址(默认本地)
base_url = http://localhost:11434

; 使用的视觉模型
model = llava

; 如果使用更大模型,可以改为:
; model = llava:13b
; model = llava:34b
; model = bakllava

; API密钥(Ollama不需要,可以随便填)
api_key = not-needed
```

### 方法2: 通过 GUI 选择(如果你的GUI支持)

在软件界面中选择:
- **视频分析模式**: Ollama本地视觉模型
- **模型名称**: llava (或其他已安装的模型)
- **服务地址**: http://localhost:11434

## 支持的所有视频分析模式

你的软件现在支持以下所有模式,客户可以自由选择:

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **gemini** | Google Gemini视频理解 | 云端分析,效果最好,需要API密钥 |
| **qwen_video** | 通义千问视频分析 | 阿里云服务,需要API密钥 |
| **frame_by_frame** | 自定义API逐帧分析 | 自己搭建的推理服务 |
| **ollama** | 本地Ollama视觉模型 | 完全本地,无需网络,免费 |
| **simulate** | 模拟模式 | 测试和演示用 |

## Ollama 模式特点

### 优势
1. **完全本地化**: 视频不会上传到任何云端,保护客户隐私
2. **零API费用**: 无需购买API密钥,降低使用成本
3. **离线可用**: 无需网络连接即可使用
4. **模型可选**: 客户可根据硬件选择不同规模的模型

### 性能
- **速度**: 逐帧分析,速度取决于硬件和模型大小
  - llava: 在普通电脑约2-5秒/帧
  - llava:13b: 约5-10秒/帧
  - llava:34b: 约10-20秒/帧

- **显存要求**:
  - llava: 约 4-6GB
  - llava:13b: 约 8-10GB
  - llava:34b: 约 20-24GB

### 建议
- 对于普通用户: 推荐 `llava` 模型
- 对于高性能电脑: 可使用 `llava:13b` 或 `llava:34b`
- 如需快速测试: 使用 `simulate` 模式

## 技术实现细节

### 工作流程
1. 视频抽帧(每隔N毫秒抽取一帧)
2. 对每一帧调用 Ollama API 进行视觉理解
3. Ollama 返回该帧的搞笑评分(0-1)和描述
4. 根据评分筛选出搞笑场景
5. 生成手绘风格插画并合成视频

### API 接口
Ollama 视觉分析使用标准的 `/api/generate` 接口:

```json
POST http://localhost:11434/api/generate
{
  "model": "llava",
  "prompt": "评估这张图片的搞笑程度...",
  "images": ["base64编码的图片"],
  "stream": false,
  "format": "json"
}
```

响应示例:
```json
{
  "response": "{\"funny_score\": 0.85, \"description\": \"人物表情夸张搞笑\"}"
}
```

### 代码位置
- 视频分析器: `universal_video_analyzer.py` 的 `OllamaVideoAnalyzer` 类
- 工作流集成: `gui_workflow.py` 的 `analyze_video_content()` 方法
- 配置读取: `gui_workflow.py` 第 353-355 行

## 故障排除

### 问题1: "Ollama服务未响应"
**解决方案**:
```bash
# 检查Ollama是否运行
curl http://localhost:11434/api/tags

# 如果失败,重启Ollama
# Windows: 在任务管理器中结束Ollama进程,然后重新打开
```

### 问题2: "Ollama模型未找到"
**解决方案**:
```bash
# 查看已安装的模型
ollama list

# 下载缺失的模型
ollama pull llava
```

### 问题3: "分析速度太慢"
**解决方案**:
1. 使用更小的模型(llava 而不是 llava:34b)
2. 增加视频抽帧间隔(config.ini 的 frame_interval)
3. 升级硬件(使用GPU加速)

### 问题4: "内存不足"
**解决方案**:
1. 使用更小的模型
2. 关闭其他占用GPU的程序
3. 减少同时分析的帧数

## 商业部署建议

### 配置建议
为不同客户提供不同配置:

**入门级客户**(普通电脑):
```ini
mode = ollama
model = llava
frame_interval = 5000  ; 每5秒抽一帧
```

**专业级客户**(高性能电脑):
```ini
mode = ollama
model = llava:13b
frame_interval = 2000  ; 每2秒抽一帧
```

**企业级客户**(云端部署):
```ini
mode = gemini  ; 或 qwen_video
```

### 销售话术
"我们的软件支持多种视频分析模式:
- **本地模式(Ollama)**: 数据不出本地,完全免费
- **云端模式(Gemini/通义)**: 分析效果最佳,按使用付费
- **自定义模式**: 对接您自己的AI服务

您可以根据需求灵活选择!"

## 总结

现在你的软件已经实现了:
✅ 多平台视频分析支持(Gemini, 通义, Ollama, 自定义)
✅ 本地离线分析能力(Ollama)
✅ 客户可自由选择模型
✅ 完整的商业化部署方案

这为你的商业软件增加了极大的灵活性和竞争力!
