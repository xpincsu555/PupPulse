# PupPulse
Real-time Emotion & Action Monitoring for Dogs

---

Table of Contents

1. [Project Overview](#project-overview--项目简介)
2. [Features](#features--功能亮点)
3. [Prerequisites](#prerequisites--前置要求)
4. [Installation](#installation--安装指南)
5. [Usage](#usage--使用说明)
6. [Project Structure](#project-structure--项目结构)
7. [License](#license--许可协议)

---

## Project Overview 

PupPulse is a real-time multimodal system that monitors the actions (sit, lie down, wag tail, etc.) and emotions (excited, relaxed, anxious, etc.) of your dog using a standard camera and microphone.

---

## Features

1. Multimodal Recognition 

   * Visual+Audio parallel processing for real-time action and emotion classification
     
2. Personalized Fine-tuning 

   * Pretrain on public datasets + auto-label your dog’s videos via ChatGPT Vision
     
3. Web Dashboard 

   * FastAPI + WebSocket with second-level updates for confidence bars & timeline
     
4. Interactive Feedback

   * Connect to feeder or smart lights for alerts or auto-treat dispensing
     
5. Extensible 

   * Extendable with 3D pose estimation, bark sentiment analysis, voice Q\&A

---

## Prerequisites

* Python 3.10+
* Miniconda / Conda
* Git & GitHub account

---

## Installation 

1. Clone the repository 

   ```bash
   git clone https://github.com/username/pup-pulse.git
   cd pup-pulse
   ```
2. Create conda environment 

   ```bash
   conda create -n pup-pulse python=3.10 -y
   conda activate pup-pulse
   ```
3. Install dependencies 

   ```bash
   pip install -r requirements.txt
   ```
4. Prepare data 

   * download data  `data/raw/`
   * `scripts/download_sample.sh` 

---

## Usage

1. **Train base model **

   ```bash
   python src/models/train_base.py --config configs/base_action.yaml
   ```
2. **Auto-label your dog **

   ```bash
   python src/utils/auto_label.py --input data/raw/mydog_videos/ --output data/labels/
   ```
3. **Fine-tune personalized model **

   ```bash
   python src/models/train_finetune.py --config configs/finetune.yaml
   ```
4. **Run real-time inference **

   ```bash
   uv run python src/api/app.py
   # 打开 http://localhost:8000/dashboard 查看仪表盘
   ```

---

## Project Structure 

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
---

## License / 许可协议

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


