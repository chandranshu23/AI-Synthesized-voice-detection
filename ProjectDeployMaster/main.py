#API
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from datetime import datetime

from model import load_model
from dataPreprocessing import (
    convert_mp3_to_wav_mono_16khz,
    spectralGatingNoiseReduction,
    create_mel_spectogram,
    standardize_spectrogram
)

import torch
from torchvision import transforms
from PIL import Image

app = FastAPI()

model = load_model("ResNet101v2_0.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# serving static files
app.mount("/static", StaticFiles(directory="static"),name="static")

# Home route
@app.get("/")
def read_root():
    return FileResponse("templates/index.html")
    # return {"hello": "world"}

# predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)

    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        wav_path = "temp.wav"
        denoised_path = "denoised.wav"
        mel_spectrogram_path = "mel_spectrogram.png"
        standardized_path = "standardized.png"

        convert_mp3_to_wav_mono_16khz(temp_filename, wav_path)
        spectralGatingNoiseReduction(wav_path, denoised_path)
        create_mel_spectogram(denoised_path, mel_spectrogram_path)
        standardize_spectrogram(mel_spectrogram_path, standardized_path)

        image = Image.open(standardized_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probability = torch.sigmoid(output).item()

        label = "AI-Generated Voice" if probability >= 0.5 else "Human Voice"

        return JSONResponse({
            "label": label,
            "confidence": round(probability, 4)
        })

    finally:
        for f in [temp_filename, wav_path, denoised_path, mel_spectrogram_path, standardized_path]:
            if os.path.exists(f):
                os.remove(f)