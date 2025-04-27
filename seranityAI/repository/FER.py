import tempfile
import os
import io
from bson import ObjectId
from moviepy.editor import VideoFileClip
from pymongo import MongoClient
import gridfs
import pandas as pd
import numpy as np
import wave
from fer import FER
from fer import Video
import matplotlib.pyplot as plt
from flask import Response, jsonify

# MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

def mget_video_file_in_memory(file_id):
    """
    Fetch video from GridFS using file_id and return a temporary file path.
    """
    try:
        file_obj = fs.get(ObjectId(file_id))
        video_bytes = file_obj.read()

        # Create a temporary file
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_file.write(video_bytes)
        temp_video_file.flush()
        temp_video_file.close()

        # Now process the video
        process_uploaded_video(temp_video_file.name)

        # After processing, optionally delete manually
        os.unlink(temp_video_file.name)

    except Exception as e:
        raise Exception(f"Error fetching video file from GridFS: {str(e)}")

def process_uploaded_video(video_path):
    try:
        print("here")
        # File paths
        audio_path = r"C:\Users\Rahma\Downloads\serenityAI- tkinter\temp_audio.wav"
        output_file = r"C:\Users\Rahma\Downloads\serenityAI- tkinter\emotions_summary_with_silence.xlsx"
        plot_output_dir = r"C:\Users\Rahma\Downloads\serenityAI- tkinter\emotion_plots"
        os.makedirs(plot_output_dir, exist_ok=True)

        silence_duration = 0

        # Load video and extract audio
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                print("Warning: No audio stream in video.")
            else:
                video_clip.audio.write_audiofile(audio_path, fps=44100, codec='pcm_s16le')

            # Silence detection
            silence_threshold = 100
            if os.path.exists(audio_path):
                with wave.open(audio_path, "r") as audio_file:
                    frame_rate = audio_file.getframerate()
                    total_frames = audio_file.getnframes()
                    channels = audio_file.getnchannels()
                    audio_data = np.frombuffer(audio_file.readframes(total_frames), dtype=np.int16)
                    audio_data = np.abs(audio_data)
                    silent_frames = np.sum(audio_data < silence_threshold)
                    silence_duration = (silent_frames / (frame_rate * channels)) / 60 if frame_rate * channels > 0 else 0
        except Exception as e:
            print(f"Error extracting or processing audio: {e}")

        # Emotion detection
        try:
            face_detector = FER(mtcnn=True)
            input_video = Video(video_path)
            processing_data = input_video.analyze(face_detector, display=False)
        except Exception as e:
            print(f"Error analyzing video for emotions: {e}")
            return

        # DataFrame operations
        try:
            vid_df = input_video.to_pandas(processing_data)
            vid_df = input_video.get_first_face(vid_df)
            vid_df = input_video.get_emotions(vid_df)
        except Exception as e:
            print(f"Error processing emotion data: {e}")
            return

        # Plot emotions
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        for i in range(0, len(emotion_columns), 2):
            emotions_to_plot = emotion_columns[i:i + 2]
            plt.figure(figsize=(20, 8))
            vid_df[emotions_to_plot].plot(fontsize=16)
            plt.title(f"{', '.join(emotions_to_plot)} over time")
            plt.savefig(os.path.join(plot_output_dir, f"plot_{i // 2 + 1}.png"))
            plt.close()

        # Save summary
        try:
            if os.path.exists(output_file):
                existing_data = pd.read_excel(output_file)
            else:
                existing_data = pd.DataFrame(columns=['Duration (minutes)', 'Silence (minutes)'])
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            existing_data = pd.DataFrame(columns=['Duration (minutes)', 'Silence (minutes)'])

        video_duration = video_clip.duration / 60
        new_data = pd.DataFrame([[video_duration, silence_duration]],
                                columns=['Duration (minutes)', 'Silence (minutes)'])
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)

        try:
            updated_data.to_excel(output_file, index=False)
            print(f"Emotions summary updated and saved to {output_file}")
        except Exception as e:
            print(f"Error saving Excel file: {e}")

    except Exception as e:
        raise Exception(f"Error processing uploaded video: {str(e)}")
