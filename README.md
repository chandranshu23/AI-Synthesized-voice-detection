
# AI-Synthesized Voice Detection 🎙️🧠

A deep learning-powered system to detect AI-generated (synthetic) speech from real human audio with high accuracy, language generalization, and real-time classification capability.

> 🛡️ Built with PyTorch, ResNet-101, FastAPI, LibROSA, and FFmpeg.

## 🚀 Project Overview

The rapid rise of text-to-speech (TTS) and voice cloning technologies poses significant threats to media trust, security, and digital communication integrity. This project addresses the challenge by building a robust voice deepfake detection system that classifies audio as either **Human** or **AI-synthesized**.

Key features:
- Real-time detection via a user-friendly web interface and FastAPI backend.
- ResNet-101-based architecture with fusion-attention modules.
- Trained on a multilingual dataset with over **96,000 audio samples**.
- Supports seven Indian languages and both genders.
- High performance: **95.76% accuracy**, **99.81% precision**, **0.999 ROC-AUC**.

## 🧠 Model Architecture

- Backbone: Modified **ResNet-101** for spectrogram-based classification.
- Input: 512×512 Mel-Frequency Spectrograms.
- Attention: Bidirectional fusion modules to capture multiscale features.
- Loss: Binary Cross-Entropy with class weighting.
- Optimizer: AdamW with learning rate scheduling.
- Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC.

## 🗂️ Dataset

- **Human Audio**: Mozilla Common Voice (English, Hindi), OpenSLR.
- **AI-Generated Audio**: Created using IIT Madras' AI4Bharat STS TTS model.
- **Languages**: English, Hindi, Kannada, Gujarati, Telugu, Tamil, Malayalam.
- **Samples**:  
  - 52,000+ Human  
  - 43,000+ AI-generated

Preprocessing:
- 16kHz mono WAV standardization using **FFmpeg**
- Spectral gating for noise/silence removal
- Data augmentation: Time-stretching, pitch-shifting, additive noise
- Spectrogram generation using **LibROSA**

## 🧰 Tech Stack

| Component      | Technology                  |
|----------------|------------------------------|
| Model Training | PyTorch, NumPy, PIL          |
| Audio Processing | LibROSA, FFmpeg             |
| Backend API    | FastAPI                      |
| Frontend       | HTML, CSS, JavaScript        |
| Deployment     | Localhost, REST API          |

## 💻 Local Setup & Usage

### 🔧 Prerequisites
- Python 3.10+
- `pip`, `virtualenv`, `git`
- FFmpeg installed and in system PATH

### 🛠️ Installation

```bash
git clone https://github.com/your-username/AI-Synthesized-voice-detection.git
cd AI-Synthesized-voice-detection/ProjectDeployMaster
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 🚦 Running the Application

```bash
uvicorn main:app --reload
```

Navigate to **http://localhost:8000** in your browser.

### 🌐 Web Interface

- Upload `.wav` or `.mp3` audio
- Audio is played back and classified
- Results shown with confidence score

## 📊 Performance

| Metric      | Value    |
|-------------|----------|
| Accuracy    | 95.76%   |
| Precision   | 99.81%   |
| Recall      | 90.87%   |
| F1 Score    | 0.9513   |
| ROC-AUC     | 0.9990   |

## 🔍 File Structure

```
ProjectDeployMaster/
├── main.py              # FastAPI backend
├── dataPreprocessing.py # Audio preprocessing (FFmpeg + LibROSA)
├── model.py             # ResNet-101 model architecture
├── templates/           # Frontend HTML
├── static/              # JS/CSS files
├── weights/             # Trained model weights
├── requirements.txt
```

## ⚠️ Challenges & Future Work

- Noise sensitivity during live microphone input
- Dataset imbalance for underrepresented languages
- Need for cross-platform deployment (beyond local)
- Future goals:
  - Add explainable AI modules
  - Support streaming audio detection
  - Build mobile-first progressive web app (PWA)

## 👨‍🏫 Academic Credit

Developed as a final-year B.Tech project under the guidance of **Dr. C. S. Negi**, College of Technology, GBPUAT, Pantnagar.
