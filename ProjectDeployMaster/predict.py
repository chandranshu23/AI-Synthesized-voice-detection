import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from datetime import datetime

from model import ResNet101v2
from dataPreprocessing import (
    convert_mp3_to_wav_mono_16khz,
    spectralGatingNoiseReduction,
    create_mel_spectogram,
    standardize_spectrogram
)

# Load model
model_path = "ResNet101v2_0.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet101v2(num_classes=1)  # FIX: match training config (binary output)

# Load checkpoint and clean keys
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["state_dict"]

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("model.", "") if k.startswith("model.") else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# 1-channel input, match model's input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_audio(audio_path):
    wav_path = "temp.wav"
    denoised_path = "denoised.wav"
    mel_spectrogram_path = "mel_spectrogram.png"
    standardized_path = "standardized.png"

    try:
        convert_mp3_to_wav_mono_16khz(audio_path, wav_path)
        spectralGatingNoiseReduction(wav_path, denoised_path)
        create_mel_spectogram(denoised_path, mel_spectrogram_path)
        standardize_spectrogram(mel_spectrogram_path, standardized_path)

        image = Image.open(standardized_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probability = torch.sigmoid(output).item()  # FIX: sigmoid for binary

        label = "AI-Generated Voice" if probability >= 0.5 else "Human Voice"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] File: {audio_path} => Prediction: {label} (Confidence: {probability:.4f})\n"

        print(log_line.strip())

        with open("predictions_log.txt", "a") as f:
            f.write(log_line)

    finally:
        for f in [wav_path, denoised_path, mel_spectrogram_path, standardized_path]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print("Error: File does not exist")
        sys.exit(1)

    predict_audio(audio_file)
