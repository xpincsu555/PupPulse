

# PupPulse

**Real-time Emotion & Action Monitoring for Your Dog**
 **面向「自家单只狗狗」的实时情绪与动作监测系统**

------

## Table of Contents

**目录**

1. [Project Overview / 项目简介](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#project-overview--项目简介)
2. [Features / 功能亮点](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#features--功能亮点)
3. [Prerequisites / 前置要求](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#prerequisites--前置要求)
4. [Installation / 安装指南](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#installation--安装指南)
5. [Usage / 使用说明](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#usage--使用说明)
6. [Project Structure / 项目结构](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#project-structure--项目结构)
7. [Contributing / 参与贡献](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#contributing--参与贡献)
8. [License / 许可协议](https://chatgpt.com/c/683491c3-dd5c-8002-9c83-39a943838d54#license--许可协议)

------

## Project Overview / 项目简介

PupPulse is a real-time multimodal system that monitors the actions (sit, lie down, wag tail, etc.) and emotions (excited, relaxed, anxious, etc.) of your dog using a standard camera and microphone backed by an RTX 3080 GPU (or similar).
 PupPulse 是一款通过普通摄像头 + 麦克风，配合 RTX 3080 GPU（或类似性能的显卡）实现对狗狗动作（坐、趴、摇尾巴等）和情绪（兴奋、放松、焦虑等）实时监测的多模态系统。

------

## Features / 功能亮点

1. **Multimodal Recognition / 多模态识别**
   - 视觉与音频并行处理，实时分类狗狗行为与情绪
   - Visual+Audio parallel processing for real-time action and emotion classification
2. **Personalized Fine-tuning / 个性化微调**
   - 基于公开数据集预训练 + ChatGPT Vision 自动标注自家狗狗
   - Pretrain on public datasets + auto-label your dog’s videos via ChatGPT Vision
3. **Web Dashboard / 网页仪表盘**
   - FastAPI + WebSocket，秒级刷新置信度条与时间轴
   - FastAPI + WebSocket with second-level updates for confidence bars & timeline
4. **Interactive Feedback / 硬件联动**
   - 对接智能投喂机或灯光，当狗狗需要互动时发通知或自动投喂
   - Connect to feeder or smart lights for alerts or auto-treat dispensing
5. **Extensible / 可拓展**
   - 支持 3D 姿态估计、吠声情感分析、语音问答
   - Extendable with 3D pose estimation, bark sentiment analysis, voice Q&A

------

## Prerequisites / 前置要求

- Python 3.10+
- Miniconda / Conda
- GPU 支持 CUDA 11.7+（可选 CPU 推理）
- Git & GitHub 账号

------

## Installation / 安装指南

1. **Clone the repository / 克隆仓库**

   ```bash
   git clone https://github.com/yourusername/pup-pulse.git
   cd pup-pulse
   ```

2. **Create conda environment / 创建 Conda 环境**

   ```bash
   conda create -n pup-pulse python=3.10 -y
   conda activate pup-pulse
   ```

3. **Install dependencies / 安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data / 准备数据**

   - 下载公开数据集至 `data/raw/`
   - （可选）先运行 `scripts/download_sample.sh` 获取示例数据

------

## Usage / 使用说明

1. **Train base model / 训练基础模型**

   ```bash
   python src/models/train_base.py --config configs/base_action.yaml
   ```

2. **Auto-label your dog / 自动标注自家狗狗**

   ```bash
   python src/utils/auto_label.py --input data/raw/mydog_videos/ --output data/labels/
   ```

3. **Fine-tune personalized model / 个性化微调**

   ```bash
   python src/models/train_finetune.py --config configs/finetune.yaml
   ```

4. **Run real-time inference / 实时推理**

   ```bash
   uv run python src/api/app.py
   # 打开 http://localhost:8000/dashboard 查看仪表盘
   ```

------

## Project Structure / 项目结构

```text
pup-pulse/
├── data/               # 原始与标注数据
├── src/                # 源代码
│   ├── models/         # 模型训练与定义
│   ├── utils/          # 数据加载、标注脚本
│   └── api/            # FastAPI 服务代码
├── requirements.txt    # Python 依赖列表
├── environment.yml     # Conda 环境导出文件
├── README.md           # 项目说明
└── .gitignore
```

------

## Contributing / 参与贡献

1. Fork 本仓库并创建分支
2. 提交 PR 并描述你的改动
3. 通过 Code Review 后合并到 `main`

欢迎提交 Issue 或 PR，让我们一起完善 PupPulse！

------

## License / 许可协议

本项目基于 MIT 协议开源，详见 [LICENSE](https://chatgpt.com/c/LICENSE)。
 This project is licensed under the MIT License. See [LICENSE](https://chatgpt.com/c/LICENSE) for details.





