import os
import sys
import shutil

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Step 1: Use environment variable or fallback to hardcoded path
ffmpeg_dir = os.getenv("FFMPEG_PATH", r"C:\Users\Rahma\Downloads\ffmpeg-7.1.1-essentials_build\bin")

# Step 2: Inject it at the beginning of PATH (before others)
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

# Step 3: Check for ffmpeg availability
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    print("❌ FFmpeg not found! PATH is:\n", os.environ["PATH"])
    raise EnvironmentError("❌ FFmpeg not found! Check the ffmpeg_dir path.")
else:
    print("✅ FFmpeg is working at:", ffmpeg_path)

import tempfile
import librosa
import pandas as pd
import whisper
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import joblib
from collections import Counter

from repository.file_repository import get_file_from_gridfs, save_excel_to_gridfs
from repository.patient_repository import (
    update_speech_excel_reference,
    get_feature_files_from_session
)


# Load models with error handling
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

try:
    print("📥 Loading Whisper model...")
    whisper_model = whisper.load_model("large-v3", device=device)
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

try:
    print("📥 Loading emotion classification models...")
    rf_model = joblib.load("best_random_forest_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    print("✅ Emotion classification models loaded successfully")
except Exception as e:
    print(f"❌ Failed to load emotion models: {e}")
    raise RuntimeError(f"Failed to load emotion classification models: {str(e)}")

CHUNK_LENGTH_SEC = 15
OFFSET_SEC = 0.6

# -----------------------------
# Augmentation Methods
# -----------------------------
def noise(data, snr_low=15, snr_high=30):
    noise_signal = np.random.normal(0, 1, size=data.shape)
    norm_constant = 2.0**15
    signal_norm = data / norm_constant
    noise_norm = noise_signal / norm_constant
    signal_power = max(np.mean(signal_norm**2), 1e-10)
    noise_power = max(np.mean(noise_norm**2), 1e-10)
    target_snr = snr_low if snr_low >= snr_high else np.random.randint(snr_low, snr_high)
    target_snr = min(target_snr, 40)
    scaling_factor = np.sqrt(signal_power / (noise_power * 10 ** (target_snr / 10)))
    return data + noise_signal * scaling_factor

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(-5, 5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def time_masking(data, sampling_rate, mask_width=20, mask_count=2):
    stft = librosa.stft(data)
    mag, phase = librosa.magphase(stft)
    for _ in range(mask_count):
        start = np.random.randint(0, mag.shape[1] - mask_width)
        mag[:, start:start+mask_width] = 0
    return librosa.istft(mag * phase)

def frequency_masking(data, sampling_rate, mask_width=10, mask_count=2):
    stft = librosa.stft(data)
    mag, phase = librosa.magphase(stft)
    for _ in range(mask_count):
        start = np.random.randint(0, mag.shape[0] - mask_width)
        mag[start:start+mask_width, :] = 0
    return librosa.istft(mag * phase)

# -----------------------------
# Helper Functions
# -----------------------------
def is_video_file(path):
    return os.path.splitext(path)[1].lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

def convert_video_to_audio(path):
    """Convert video to audio with proper resource cleanup"""
    video = None
    try:
        video = VideoFileClip(path)
        if video.audio is None:
            raise ValueError("Video file has no audio track")
            
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = temp_audio_file.name
        video.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
        return audio_path
    except Exception as e:
        raise Exception(f"Failed to convert video to audio: {str(e)}")
    finally:
        # Ensure video resources are cleaned up
        if video is not None:
            try:
                if hasattr(video, 'reader') and video.reader:
                    video.reader.close()
                if hasattr(video, 'audio') and video.audio:
                    video.audio.reader.close_proc()
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Failed to cleanup video resources: {cleanup_error}")

def convert_to_wav(input_path):
    """Convert audio file to WAV format"""
    try:
        audio = AudioSegment.from_file(input_path)
        wav_path = input_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise Exception(f"Failed to convert audio to WAV: {str(e)}")

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(signal, sr, flatten=True, frame_length=2048, hop_length=512):
    def zcr(data): return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length, hop_length))
    def rmse(data): return np.squeeze(librosa.feature.rms(data, frame_length, hop_length))
    def mfcc(data, sr): return np.ravel(librosa.feature.mfcc(data, sr, n_mfcc=13).T) if flatten else np.squeeze(librosa.feature.mfcc(data, sr, n_mfcc=13).T)
    def mel(data, sr):
        mel = librosa.feature.melspectrogram(data, sr=sr, n_fft=1024, win_length=512, window="hamming", hop_length=256, n_mels=100, fmax=sr / 2)
        return np.ravel(librosa.amplitude_to_db(mel, ref=np.max).T) if flatten else np.squeeze(librosa.amplitude_to_db(mel, ref=np.max).T)

    return np.hstack([zcr(signal), rmse(signal), mfcc(signal, sr), mel(signal, sr)])

# -----------------------------
# Main Analysis Function
# -----------------------------
def run_speech_tov_on_video(file_id, patient_id, session_id):
    """
    Run speech and tone-of-voice analysis on video file with minimal error handling
    
    Args:
        file_id: GridFS file ID of video
        patient_id: Patient identifier  
        session_id: Session identifier
        
    Returns:
        tuple: (dataframe, excel_file_id)
        
    Raises:
        Exception: If analysis fails with error details
    """
    try:
        # Input validation
        if not file_id or not patient_id or not session_id:
            raise ValueError("file_id, patient_id, and session_id are required")
            
        print(f"🎙️ Starting speech analysis for patient {patient_id}, session {session_id}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "video.mp4")
            
            try:
                get_file_from_gridfs(file_id, path)
            except Exception as e:
                raise Exception(f"Failed to download video from GridFS: {str(e)}")

            # Verify file exists and has content
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise Exception("Downloaded video file is empty or doesn't exist")

            print(f"✅ Video downloaded: {os.path.getsize(path) / 1024 / 1024:.2f}MB")

            # Convert video to audio if needed
            if is_video_file(path):
                print("Converting video to audio for emotions...")
                try:
                    path = convert_video_to_audio(path)
                except Exception as e:
                    raise Exception(f"Video to audio conversion failed: {str(e)}")

            # Convert to WAV if needed
            if not path.lower().endswith(".wav"):
                print("Converting audio to WAV format for emotions...")
                try:
                    path = convert_to_wav(path)
                except Exception as e:
                    raise Exception(f"Audio format conversion failed: {str(e)}")

            # Double-check FFmpeg availability
            if not shutil.which("ffmpeg"):
                raise EnvironmentError("❌ FFmpeg not found in PATH during Whisper transcription!")

            # Transcribe audio
            print("Transcribing audio...")
            try:
                result = whisper_model.transcribe(path, task="transcribe")
            except Exception as e:
                raise Exception(f"Audio transcription failed: {str(e)}")

            if not result or "text" not in result:
                raise Exception("No transcription result obtained from audio")

            text = result["text"]
            if not text or len(text.strip()) == 0:
                raise Exception("Empty transcription - audio may have no speech content")

            # Process transcription segments
            interval_texts = []
            current_text = []
            current_start = 0

            segments = result.get("segments", [])
            if not segments:
                print("⚠️ No segment information available, using full text")
                interval_texts = [text]
            else:
                for segment in segments:
                    while segment["start"] >= current_start + CHUNK_LENGTH_SEC:
                        interval_texts.append("".join(current_text))
                        current_start += CHUNK_LENGTH_SEC
                        current_text = []
                    current_text.append(segment["text"] + " ")
                if current_text:
                    interval_texts.append("".join(current_text))

            # Load audio data for emotion analysis
            print("Loading audio data for emotion analysis...")
            try:
                data, sr = librosa.load(path)
            except Exception as e:
                raise Exception(f"Failed to load audio data: {str(e)}")

            if len(data) == 0:
                raise Exception("Audio file contains no data")

            # Process audio segments
            segment_length = CHUNK_LENGTH_SEC * sr
            offset_samples = int(OFFSET_SEC * sr)

            result_chunks = [data[i+offset_samples:i+segment_length] 
                           for i in range(0, len(data), segment_length) 
                           if len(data[i+offset_samples:i+segment_length]) > 0]

            if not result_chunks:
                raise Exception("No audio chunks could be extracted for analysis")

            print(f"Processing {len(result_chunks)} audio segments...")
            all_data = []
            predictions = []

            for i, segment in enumerate(result_chunks):
                if len(segment) == 0 or np.max(np.abs(segment)) == 0:
                    predictions.append("Unknown")
                    continue

                try:
                    segment = segment / np.max(np.abs(segment))
                    signal = np.zeros(int(sr * 6))
                    signal[:min(len(signal), len(segment))] = segment[:min(len(signal), len(segment))]

                    features = np.vstack([
                        extract_features(signal, sr),
                        extract_features(noise(signal), sr),
                        extract_features(pitch(signal, sr), sr),
                        extract_features(noise(pitch(signal, sr), sr), sr),
                        extract_features(time_masking(signal, sr), sr),
                        extract_features(frequency_masking(signal, sr), sr)
                    ])
                    X_real_scaled = scaler.transform(features.astype(np.float32))
                    pred = Counter(rf_model.predict(X_real_scaled)).most_common(1)[0][0]
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error processing segment {i+1}: {e}")
                    predictions.append("Unknown")

            # Ensure we have equal length arrays
            min_length = min(len(interval_texts), len(predictions))
            interval_texts = interval_texts[:min_length]
            predictions = predictions[:min_length]

            # Build final data
            for i, (transcription, prediction) in enumerate(zip(interval_texts, predictions)):
                all_data.append([i + 1, transcription.strip(), prediction])

            if not all_data:
                raise Exception("No analysis data could be generated")

            # Save results
            df = pd.DataFrame(all_data, columns=["Chunk Number", "Transcription", "Prediction"])
            excel_path = os.path.join(tmpdir, f"{patient_id}_{session_id}_speech_tov_chunks.xlsx")
            
            try:
                df.to_excel(excel_path, index=False)
                file_id = save_excel_to_gridfs(excel_path, os.path.basename(excel_path))
                update_speech_excel_reference(patient_id, session_id, file_id, text)
            except Exception as e:
                raise Exception(f"Failed to save analysis results: {str(e)}")

            print(f"✅ Speech analysis completed: {len(all_data)} chunks processed")
            return df, file_id

    except Exception as e:
        error_msg = f"❌ Speech analysis failed: {str(e)}"
        print(error_msg)
        raise Exception({"error": "Speech analysis failed", "details": str(e)})

def process_speech_and_tov_by_id(file_id, patient_id, session_id):
    """
    Process speech and TOV analysis and return results as dictionary records
    
    Args:
        file_id: GridFS file ID of video
        patient_id: Patient identifier  
        session_id: Session identifier
        
    Returns:
        list: List of dictionaries containing analysis results
        
    Raises:
        Exception: If analysis fails
    """
    try:
        df, _ = run_speech_tov_on_video(file_id, patient_id, session_id)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"❌ process_speech_and_tov_by_id failed: {str(e)}")
        raise