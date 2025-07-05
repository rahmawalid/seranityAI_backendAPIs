import os
import matplotlib

os.environ["MPLBACKEND"] = "Agg"

import tempfile
import shutil
import pandas as pd
from fer import FER, Video

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from repository.file_repository import (
    get_file_from_gridfs,
    save_excel_to_gridfs,
    save_image_to_gridfs,
)
from repository.patient_repository import (
    update_fer_excel_reference,
    update_fer_plot_reference,
)


def run_fer_on_video_by_id(file_id, patient_id, session_id):
    """
    Run FER analysis on video file with minimal error handling
    
    Args:
        file_id: GridFS file ID of video
        patient_id: Patient identifier  
        session_id: Session identifier
        
    Returns:
        tuple: (dataframe, excel_file_id, plot_ids)
        
    Raises:
        Exception: If analysis fails with error details
    """
    input_video = None
    
    try:
        # Input validation
        if not file_id or not patient_id or not session_id:
            raise ValueError("file_id, patient_id, and session_id are required")
            
        print("🔍 Step 1: Downloading video from GridFS")

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            
            try:
                get_file_from_gridfs(file_id, video_path)
            except Exception as e:
                raise Exception(f"Failed to download video from GridFS: {str(e)}")

            # Verify file exists and has content
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Downloaded video file is empty or doesn't exist")

            print(f"✅ Step 2: Video downloaded to {video_path}")
            print(f"📦 File size (MB): {os.path.getsize(video_path) / 1024 / 1024:.6f}")

            # Step 3: Run FER analysis
            print("⚙️ Step 3: Running FER analysis on sampled frames...")
            
            try:
                input_video = Video(video_path)
                detector = FER(mtcnn=True)
                data = input_video.analyze(detector, display=False)
                df = input_video.to_pandas(data)
            except Exception as e:
                raise Exception(f"FER analysis failed: {str(e)}")

            # Verify we got data
            if df is None or df.empty:
                raise Exception("No emotion data extracted from video - video may be corrupted or too short")

            print("📊 Step 4: Building DataFrame")
            excel_path = os.path.join(tmpdir, f"{patient_id}_{session_id}_fer.xlsx")
            
            try:
                df.to_excel(excel_path, index=False)
                excel_file_id = save_excel_to_gridfs(
                    excel_path, os.path.basename(excel_path)
                )
                update_fer_excel_reference(patient_id, session_id, excel_file_id)
            except Exception as e:
                raise Exception(f"Failed to save Excel file: {str(e)}")

            print("🖼️ Step 5: Generating plots")
            plot_ids = []
            available_emotions = df.columns.intersection(
                ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            )

            if available_emotions.empty:
                print("⚠️ No emotion columns found in the DataFrame. Skipping plot generation.")
            else:
                try:
                    for i in range(0, len(available_emotions), 2):
                        emotions_to_plot = available_emotions[i : i + 2]
                        
                        try:
                            ax = df[emotions_to_plot].plot(figsize=(20, 8), fontsize=16)
                            plt.title(f"{', '.join(emotions_to_plot)} over time", fontsize=20)

                            plot_filename = f"plot_{i//2 + 1}.png"
                            plot_path = os.path.join(tmpdir, plot_filename)
                            plt.savefig(plot_path)
                            plt.close()

                            plot_id = save_image_to_gridfs(
                                plot_path,
                                f"{patient_id}_{session_id}_{'_'.join(emotions_to_plot)}.png",
                            )
                            plot_ids.append(plot_id)
                        except Exception as e:
                            print(f"⚠️ Failed to generate plot for {emotions_to_plot}: {str(e)}")
                            plt.close()  # Ensure plot is closed even if save fails
                            continue

                    if plot_ids:  # Only update if we have plots
                        update_fer_plot_reference(patient_id, session_id, plot_ids)
                    
                except Exception as e:
                    print(f"⚠️ Plot generation had issues: {str(e)}")
                    # Continue without plots - analysis data is still valid

            return df, excel_file_id, plot_ids

    except Exception as e:
        error_msg = f"❌ FER analysis failed: {str(e)}"
        print(error_msg)
        raise Exception({"error": "FER analysis failed", "details": str(e)})
    
    finally:
        # Ensure video resources are cleaned up
        if input_video is not None:
            try:
                if hasattr(input_video, "reader") and input_video.reader:
                    input_video.reader.close()
                if hasattr(input_video, "audio") and input_video.audio:
                    input_video.audio.reader.close_proc()
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Failed to cleanup video resources: {cleanup_error}")