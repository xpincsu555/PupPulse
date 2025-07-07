# PupPulse
<img src="assets/PupPulse.png" alt="PupPulse Logo" width="300"/>
Real-time Emotion & Action Monitoring for Dogs

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [License](#License)

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
 
**1. Clone the repository**

   ```bash
   git clone https://github.com/xpincsu555/PupPulse.git
   cd PupPulse
   ```

**2. Install Kaggle CLI & configure API token**

   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/    # 把下载的 kaggle.json 放到这里
   chmod 600 ~/.kaggle/kaggle.json
   ```

**3. Create & Activate Conda Environment**

   ```bash
   conda create -n pup-pulse python=3.10 -y
   conda activate pup-pulse
   ```
**4. Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
**5. Download & Prepare data**

  ```bash
   # 1) Download into data/raw
   kaggle datasets download -d devzohaib/dog-emotions-prediction \
      -p data/raw

   # 2) Unzip into data/raw/images
   mkdir -p data/raw/images
   unzip -o data/raw/dog-emotions-prediction.zip -d data/raw/images

   # 3) Verify the four folders exist
   ls data/raw/images
   # → angry  happy  relaxed  sad
   ```


---

## Usage

1. **Train the emotion‐classification model**

   ```bash
   python src/models/train.py \
     --config configs/emotion_classification.yaml
   ```
2. **Evaluate on the validation set**

   ```bash
  python src/models/evaluate.py \
  --weights outputs/best_emotion_model.pth
   ```
3. **Run inference on a single image**

   ```bash
   python src/inference/predict.py \
  --image data/raw/images/happy/sample.jpg
   ```

4. **Auto‐label your own videos**  
   ```bash
   python src/utils/auto_label.py \
     --input data/raw/mydog_videos/ \
     --output data/labels/
   ``` 

5.**Fine-tune on personalized labels**
   ```bash
   python src/models/train_finetune.py \
  --config configs/finetune.yaml
  ```


---

## Project Structure

```text
PupPulse/
├── assets/             # Images and logos
├── configs/
├── data/               # Raw and annotated data
│   └── raw/
│       └── images/
│           ├── angry/
│           ├── happy/
│           ├── relaxed/
│           └── sad/
├── src/                # Source code
│   ├── models/         # Model definitions & training scripts
│   ├── utils/          # Data loading & labeling scripts
│   └── inference/
├── scripts/
│   └── unzip_data.sh
├── requirements.txt    # Python dependencies
├── environment.yml     # Conda environment file
├── README.md           # Project documentation
├── LICENSE
└── .gitignore
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


