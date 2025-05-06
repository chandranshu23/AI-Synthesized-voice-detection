import librosa
print(librosa.__version__)
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os
import glob
import scipy
import numpy as np
import subprocess
from PIL import Image
import shutil


# FFMPEG_PATH = shutil.which("ffmpeg") or r"C:\Users\acer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
FFMPEG_PATH = shutil.which("ffmpeg")

#Fallback for running locally and not found
if FFMPEG_PATH is None and os.name == "nt":
    FFMPEG_PATH = r"C:\Users\acer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"

#raising error if still not found
if not FFMPEG_PATH or not os.path.isfile(FFMPEG_PATH):
    raise RuntimeError(f"ffmpeg not found. Please ensure it is installed and in your system PATH.")

# if not os.path.isfile(FFMPEG_PATH):
#     raise RuntimeError(f"ffmpeg.exe not found at {FFMPEG_PATH}, please install or correct the path.")

def fetch_audio_files(folder_path):
    # Define the audio file extensions we have
    audio_extensions = ['*.mp3', '*.wav']
    audio_files = []  # empty list to store all audio file paths
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))  # Loop through each extension and fetch the files
    return audio_files

def convert_mp3_to_wav_mono_16khz(input_file, output_file):

    cmd = [
        FFMPEG_PATH,
        "-y",              # overwrite if exists
        "-i", input_file,  # input file
        "-ar", "16000",    # sample rate
        "-ac", "1",        # mono
        output_file
    ]
    print("Running FFmpeg:", " ".join(cmd))
    # Convert audio using FFmpeg first
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed for {input_file}:\n{e.stderr.decode()}")
        return

    # Load the FFmpeg-converted file with librosa
    try:
        y, sr = librosa.load(output_file, sr=16000)
        sf.write(output_file, y, samplerate=16000, subtype="PCM_16")
    except Exception as e:
        print(f"Librosa failed to load {output_file}: {e}")

def spectralGatingNoiseReduction(input_file, output_file, noise_duration=0.5, sr=16000):
    try:
        audio, sr = librosa.load(input_file, sr=sr)
    except Exception as e:
        print(f"Error loading {input_file} in spectralGatingNoiseReduction: {e}")
        raise
    stft = librosa.stft(audio)  # compute short-time Fourier Transform (stft)
    magnitude, phase = librosa.magphase(stft)  # get magnitude and phase
    noise_stft = np.mean(magnitude[:, :int(sr * noise_duration)], axis=1, keepdims=True)  # estimate noise
    cleaned_stft = np.maximum(magnitude - noise_stft, 0) * phase  # reduce noise
    cleaned_audio = librosa.istft(cleaned_stft)  # convert back to time-domain
    sf.write(output_file, cleaned_audio, samplerate=16000)

def augument_audio(input_file, output_file, stretch_rate=1.1, pitch_steps=2, sr=16000):
    try:
        audio, sr = librosa.load(input_file, sr=sr)
    except Exception as e:
        print(f"Error loading {input_file} in augument_audio: {e}")
        raise
    stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_rate)  # time stretching
    shifted_audio = librosa.effects.pitch_shift(stretched_audio, sr=sr, n_steps=pitch_steps)  # pitch shifting
    sf.write(output_file, shifted_audio, samplerate=16000)

def create_mel_spectogram(input_file, output_image, sr=16000, n_mels=128, dpi=300):
    try:
        audio, sr = librosa.load(input_file, sr=sr)
    except Exception as e:
        print(f"Error loading {input_file} in create_mel_spectogram: {e}")
        raise
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)  # compute Mel Spectrogram
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # convert to decibels
    # Plot and save the spectrogram as an image
    fig, ax = plt.subplots(figsize=(1.7, 1.7), dpi=dpi)
    ax.set_axis_off()  # remove axis for clean image input
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis=None, y_axis=None, cmap='magma')
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close()

def standardize_spectrogram(input_image, output_image, size=(512,512)):
    # Convert spectrogram image to standard size and single channel
    img = Image.open(input_image).convert("L")
    img_resized = img.resize(size)
    img_resized.save(output_image)