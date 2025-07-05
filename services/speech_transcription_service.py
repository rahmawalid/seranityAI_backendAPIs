import os
import tempfile
import shutil
import torch
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import whisper
import google.generativeai as genai
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from bson import ObjectId

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from config import fs
from repository.file_repository import get_file_from_gridfs
from repository.patient_repository import update_transcription_pdf_reference

# ================================
# CONFIGURATION AND SETUP
# ================================

# Use environment variables with fallbacks
FFMPEG_DIR = os.getenv("FFMPEG_PATH", r"C:\Users\Rahma\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin")
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBsDcl5tRJd6FR0fy0pNvwv76-S5QrVvK4')

# Constants
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"]
AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac", ".flac"]

# PDF layout constants
PDF_MARGIN_RIGHT = 550
PDF_START_Y = 750
PDF_LINE_HEIGHT = 20
PDF_MAX_WIDTH = 500

# Font paths with environment variable option
ARABIC_FONT_PATHS = [
    os.path.join(os.path.dirname(__file__), "Amiri-Regular.ttf"),  # Same directory as this file
    os.path.abspath("Amiri-Regular.ttf"),  # Project root
    "Amiri-Regular.ttf"  # Relative path
]

ARABIC_BOLD_FONT_PATHS = [
    os.path.join(os.path.dirname(__file__), "Amiri-Bold.ttf"),  # Same directory as this file
    os.path.abspath("Amiri-Bold.ttf"),  # Project root
    "Amiri-Bold.ttf"  # Relative path
]

# Whisper model path (can be environment variable or let Whisper download)
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", None)  # If None, Whisper will download

def _setup_ffmpeg():
    """Setup FFmpeg with error handling"""
    try:
        if FFMPEG_DIR not in os.environ["PATH"]:
            os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]
        
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise EnvironmentError("❌ FFmpeg not found! Check the FFMPEG_PATH environment variable or ffmpeg installation.")
        
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
        return ffmpeg_path
    except Exception as e:
        print(f"❌ FFmpeg setup failed: {e}")
        raise

def _setup_whisper_model():
    """Setup Whisper model with teammate's approach but with fallback"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    try:
        # Try teammate's approach first (local model path)
        if WHISPER_MODEL_PATH and os.path.exists(WHISPER_MODEL_PATH):
            print(f"📥 Loading Whisper model from: {WHISPER_MODEL_PATH}")
            model = whisper.load_model(WHISPER_MODEL_PATH, device=device)
            print("✅ Local Whisper model loaded successfully")
            return model
        else:
            # Fallback to downloading large-v3-turbo (teammate's preference) 
            print("📥 Loading Whisper large-v3 model...")
            try:
                model = whisper.load_model("large-v3", device=device)
                print("✅ Whisper large-v3 model loaded successfully")
                return model
            except Exception as e:
                print(f"⚠️ Error loading large-v3 model: {e}")
                # Final fallback to base
                print("📥 Falling back to base model...")
                model = whisper.load_model("base", device=device)
                print("✅ Whisper base model loaded as fallback")
                return model
                
    except Exception as e:
        print(f"❌ Error loading any Whisper model: {e}")
        raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

def _setup_gemini():
    """Setup Gemini model with error handling"""
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not found in environment variables")
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Gemini model configured successfully")
        return model
    except Exception as e:
        print(f"⚠️ Error setting up Gemini: {e}")
        return None

# Initialize components with error handling
try:
    _setup_ffmpeg()
    WHISPER_MODEL = _setup_whisper_model()
    GEMINI_MODEL = _setup_gemini()
    print("✅ All services initialized successfully")
except Exception as e:
    print(f"❌ Service initialization failed: {e}")
    raise

# ================================
# UTILITY FUNCTIONS
# ================================

def _is_video_file(file_path):
    """Check if file is a video file"""
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS

def _is_audio_file(file_path):
    """Check if file is an audio file"""
    return os.path.splitext(file_path)[1].lower() in AUDIO_EXTENSIONS

def _get_safe_filename(patient_id, session_id, suffix=""):
    """Generate safe filename for output files"""
    return f"transcription_{patient_id}_{session_id}{suffix}"

def _cleanup_temp_files(*file_paths):
    """Clean up temporary files safely"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"⚠️ Could not clean up {file_path}: {e}")

# ================================
# MEDIA PROCESSING FUNCTIONS (Using teammate's logic)
# ================================

def _convert_video_to_audio(video_path, output_dir):
    """Convert video to audio with proper resource management (teammate's method)"""
    print("🎬 Converting video to audio...")
    video = None
    
    try:
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            raise ValueError("Video file has no audio track")
        
        # Use teammate's approach - NamedTemporaryFile 
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = temp_audio_file.name
        
        video.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
        
        print(f"✅ Video converted to audio: {audio_path}")
        return audio_path
        
    except Exception as e:
        print(f"❌ Error converting video to audio: {e}")
        raise Exception(f"Video to audio conversion failed: {str(e)}")
    finally:
        # Ensure video resources are cleaned up
        if video is not None:
            try:
                if hasattr(video, 'close'):
                    video.close()
                if hasattr(video, 'audio') and video.audio:
                    video.audio.close()
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Failed to cleanup video resources: {cleanup_error}")

def _convert_to_wav(input_path):
    """Convert audio to WAV format (teammate's approach)"""
    print("🎵 Converting audio to WAV format...")
    
    try:
        audio = AudioSegment.from_file(input_path)
        wav_path = input_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
        
        print(f"✅ Audio converted to WAV: {wav_path}")
        return wav_path
        
    except Exception as e:
        print(f"❌ Error converting to WAV: {e}")
        raise Exception(f"Audio format conversion failed: {str(e)}")

def _process_media_file(input_path, temp_dir):
    """Process media file with validation and error handling (teammate's logic)"""
    print(f"🔄 Processing media file: {os.path.basename(input_path)}")
    
    try:
        # Verify file exists and has content
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise ValueError("Input file is empty or doesn't exist")
        
        # Process based on file type (teammate's approach)
        if _is_video_file(input_path):
            processed_path = _convert_video_to_audio(input_path, temp_dir)
        elif _is_audio_file(input_path):
            processed_path = input_path
        else:
            ext = os.path.splitext(input_path)[1]
            raise ValueError(f"Unsupported file type: {ext}")

        # Ensure output is WAV format (teammate's approach)
        if not processed_path.lower().endswith(".wav"):
            processed_path = _convert_to_wav(processed_path)
        
        return processed_path
        
    except Exception as e:
        raise Exception(f"Media file processing failed: {str(e)}")

# ================================
# TRANSCRIPTION FUNCTIONS (Using teammate's enhanced approach)
# ================================

def _transcribe_audio(audio_path):
    """Transcribe audio using teammate's approach with error handling"""
    print("🎯 Transcribing...")
    
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise ValueError("Audio file is empty or doesn't exist")
        
        # Use teammate's transcription approach
        result = WHISPER_MODEL.transcribe(audio_path, task="transcribe")
        
        if not result or "segments" not in result:
            raise ValueError("No transcription result obtained from audio")
        
        # Extract transcription text (teammate's method)
        segments = result.get("segments", [])
        if not segments:
            raise ValueError("No segments found in transcription - audio may be too short or silent")
        
        # Build transcription string (teammate's approach)
        transcription_string = ""
        for i in range(len(segments)):
            transcription_string += segments[i]['text'] + "\n\n"
        
        transcription = transcription_string.strip()
        if not transcription:
            raise ValueError("Empty transcription result - audio may contain no speech")
        
        print(f"✅ Transcription completed. Length: {len(transcription)} characters")
        return transcription
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        raise Exception(f"Audio transcription failed: {str(e)}")

def _process_with_gemini(raw_transcription):
    """Process transcription with Gemini AI using teammate's enhanced prompt"""
    if not GEMINI_MODEL:
        print("⚠️ Gemini not available, returning raw transcription")
        return raw_transcription
    
    print("🤖 Generating grammar correction...")
    
    # Use teammate's enhanced prompt for Egyptian Arabic therapy sessions
    prompt = f"""
    You are given a raw, unpunctuated therapy session transcript in Egyptian Arabic, formatted as a sequence of text segments, each with an ID. Each segment contains spoken text from either the patient or the therapist, but the speaker is not labeled.

    Your task is to:
    1. Add punctuation and correct grammar without changing the original meaning.
    2. Identify and label speakers based on context. Use:
       - الدكتور: for the therapist
       - المريض: for the client/patient (regardless of name)
    3. Ensure pronouns and verbs match the speaker's gender and context.
    4. Do not add, invent, or remove information unless a very short word is clearly required to make the sentence understandable.
    5. Do not write "segment", "segment 1", or similar labels — just use the speaker label for each line of text.
    6. Do not remove or merge lines, even if the same speaker is continuing. Always preserve the 1:1 mapping with input segments.
    7. Fix any clear ASR (automatic speech recognition) errors if the correct meaning is obvious — such as missing connecting words or common misheard phrases.
    8. Only add small connecting words if required for meaning. Do not introduce new content.
    9. Ensure the correct gender is used for verbs and pronouns.
    10. Output in this format:

    <Speaker>: <cleaned and corrected text>

    Now process the following transcription lines:

    {raw_transcription}
    """
    
    try:
        # Use teammate's temperature setting
        response = GEMINI_MODEL.generate_content(
            prompt, 
            generation_config={"temperature": 0.0}
        )
        
        if not response or not response.text:
            print("⚠️ Gemini returned empty response, using raw transcription")
            return raw_transcription
        
        refined_transcription = response.text.strip()
        print("✅ Gemini processing completed")
        return refined_transcription
        
    except Exception as e:
        print(f"⚠️ Gemini processing failed: {e}, returning raw transcription")
        return raw_transcription

# ================================
# PDF GENERATION (Using teammate's Arabic-focused approach)
# ================================

def _setup_fonts():
    """Enhanced font setup with better Arabic support detection"""
    fonts_config = {
        "title_font": "Helvetica-Bold",      # Fallback
        "body_font": "Helvetica",            # Fallback  
        "header_font": "Helvetica-Bold",     # Fallback
        "supports_arabic": False
    }
    
    print("🔍 Looking for Amiri fonts in project directory...")
    
    # Try to register Amiri Regular
    amiri_regular_registered = False
    amiri_bold_registered = False
    
    for arabic_font_path in ARABIC_FONT_PATHS:
        print(f"   Trying: {arabic_font_path}")
        if os.path.exists(arabic_font_path):
            try:
                pdfmetrics.registerFont(TTFont("Amiri", arabic_font_path))
                amiri_regular_registered = True
                print(f"✅ Amiri Regular registered: {arabic_font_path}")
                break
            except Exception as e:
                print(f"⚠️ Failed to register Amiri Regular {arabic_font_path}: {e}")
                continue
        else:
            print(f"   File not found: {arabic_font_path}")
    
    # Try to register Amiri Bold
    for arabic_bold_path in ARABIC_BOLD_FONT_PATHS:
        print(f"   Trying Bold: {arabic_bold_path}")
        if os.path.exists(arabic_bold_path):
            try:
                pdfmetrics.registerFont(TTFont("Amiri-Bold", arabic_bold_path))
                amiri_bold_registered = True
                print(f"✅ Amiri Bold registered: {arabic_bold_path}")
                break
            except Exception as e:
                print(f"⚠️ Failed to register Amiri Bold {arabic_bold_path}: {e}")
                continue
        else:
            print(f"   Bold file not found: {arabic_bold_path}")
    
    # Configure fonts based on what was registered
    if amiri_regular_registered:
        if amiri_bold_registered:
            # Both fonts available - optimal configuration
            fonts_config.update({
                "title_font": "Amiri-Bold",
                "body_font": "Amiri", 
                "header_font": "Amiri-Bold",
                "supports_arabic": True
            })
            print("✅ Using Amiri Bold + Regular with full RTL support")
        else:
            # Only regular available
            fonts_config.update({
                "title_font": "Amiri",
                "body_font": "Amiri", 
                "header_font": "Amiri",
                "supports_arabic": True
            })
            print("✅ Using Amiri Regular with RTL support (no bold)")
    else:
        print("❌ No Amiri fonts found, using Helvetica fallback (limited Arabic support)")
    
    return fonts_config

def _format_text_unified(text):
    """Enhanced unified text formatting with better Arabic handling"""
    try:
        if not text:
            return ""
        
        text = str(text).strip()
        
        # Check if text contains Arabic characters
        has_arabic = any('\u0600' <= ch <= '\u06ff' for ch in text)
        
        if has_arabic:
            # Apply Arabic text shaping and bidirectional algorithm
            try:
                # First reshape for proper Arabic glyph connection
                reshaped = reshape(text)
                # Then apply bidirectional algorithm for proper RTL layout
                bidi_text = get_display(reshaped)
                return bidi_text
            except Exception as e:
                print(f"⚠️ Arabic reshaping failed: {e}")
                return text
        else:
            # English text - return as-is but ensure it works with RTL layout
            return text
            
    except Exception as e:
        print(f"⚠️ Text formatting failed: {e}")
        return str(text)

# Keep the old function name for compatibility but redirect to new one
def _format_arabic_text(text):
    """Legacy function - redirects to unified formatter"""
    return _format_text_unified(text)

def _add_pdf_header(canvas_obj, page_number, patient_id, session_id, font_config):
    """Add header to PDF with consistent RTL alignment for Arabic"""
    try:
        canvas_obj.setFont(font_config["header_font"], 14)
        
        header_text = f"Therapy Session Transcription – Patient {patient_id} – Session {session_id}"
        page_text = f"Page {page_number}"
        
        # Format text
        formatted_header = _format_text_unified(header_text)
        formatted_page = _format_text_unified(page_text)
        
        # For mixed content headers, check if we should use RTL layout
        header_has_arabic = any('\u0600' <= ch <= '\u06ff' for ch in formatted_header)
        
        if font_config["supports_arabic"]:
            # Use RTL layout for Arabic-capable fonts
            canvas_obj.drawRightString(550, 780, formatted_header)
            canvas_obj.drawString(50, 780, formatted_page)
        else:
            # Fallback for non-Arabic fonts
            canvas_obj.drawString(50, 780, formatted_header)
            canvas_obj.drawRightString(550, 780, formatted_page)
            
    except Exception as e:
        print(f"⚠️ Error adding PDF header: {e}")

def _draw_pdf_content(canvas_obj, content, font_config):
    """Draw content with consistent RTL alignment for Arabic text"""
    try:
        lines = content.strip().splitlines()
        page_number = 1
        y = 670
        
        # Add header
        _add_pdf_header(canvas_obj, page_number, 
                       font_config.get("patient_id", "UNKNOWN"), 
                       font_config.get("session_id", "UNKNOWN"), 
                       font_config)
        
        for line in lines:
            if not line.strip():
                continue
            
            try:
                if ":" in line and len(line.split(":", 1)) == 2:
                    speaker, message = line.split(":", 1)
                    
                    # Speaker name (always right-aligned for Arabic context)
                    canvas_obj.setFont(font_config["header_font"], 12)
                    speaker_text = _format_text_unified(speaker + ":")
                    
                    # Always use RTL alignment when Arabic fonts are available
                    if font_config["supports_arabic"]:
                        canvas_obj.drawRightString(550, y, speaker_text)
                    else:
                        # Fallback for systems without Arabic fonts
                        canvas_obj.drawString(50, y, speaker_text)
                    
                    y -= 20
                    
                    # Message content (always right-aligned for Arabic context)
                    canvas_obj.setFont(font_config["body_font"], 11)
                    message_lines = _wrap_text_for_pdf(message.strip(), font_config)
                    
                    for formatted_line in message_lines:
                        if font_config["supports_arabic"]:
                            # Always right-align content in Arabic context
                            canvas_obj.drawRightString(530, y, formatted_line)  # Slightly indented from margin
                        else:
                            # Fallback left alignment
                            canvas_obj.drawString(70, y, formatted_line)  # Indented from left
                        y -= 20
                else:
                    # Handle lines without speaker designation
                    canvas_obj.setFont(font_config["body_font"], 11)
                    formatted_line = _format_text_unified(line)
                    
                    if font_config["supports_arabic"]:
                        # Right-align all content
                        canvas_obj.drawRightString(550, y, formatted_line)
                    else:
                        # Left-align fallback
                        canvas_obj.drawString(50, y, formatted_line)
                    y -= 20
                
                y -= 5  # Extra spacing between speaker blocks
                
                # New page if needed
                if y < 100:
                    canvas_obj.showPage()
                    page_number += 1
                    _add_pdf_header(canvas_obj, page_number, 
                                   font_config.get("patient_id", "UNKNOWN"), 
                                   font_config.get("session_id", "UNKNOWN"), 
                                   font_config)
                    y = 670
                    
            except Exception as e:
                print(f"⚠️ Error drawing line '{line[:50]}...': {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error drawing PDF content: {e}")
        raise
def _wrap_text_for_pdf(text, font_config, max_width=400):
    """Wrap text for PDF with proper Arabic handling"""
    try:
        if not text or not text.strip():
            return []
        
        # Format the text first
        formatted_text = _format_text_unified(text)
        
        # Simple word-based wrapping (can be enhanced with proper font metrics)
        words = formatted_text.split()
        lines = []
        current_line = ""
        
        # Estimate characters per line (adjust based on font size and page width)
        chars_per_line = 60  # Conservative estimate for 11pt font
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            
            if len(test_line) <= chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word too long, force it
                    lines.append(word)
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [formatted_text]
        
    except Exception as e:
        print(f"⚠️ Text wrapping error: {e}")
        return [_format_text_unified(text)]

def _generate_pdf(content, pdf_path, patient_id, session_id):
    """Generate PDF with consistent RTL alignment for Arabic content"""
    print(f"📄 Generating PDF with RTL alignment: {pdf_path}")
    
    try:
        if not content or not content.strip():
            raise ValueError("Cannot generate PDF from empty content")
        
        # Setup fonts
        font_config = _setup_fonts()
        font_config["patient_id"] = patient_id
        font_config["session_id"] = session_id
        
        # Create PDF canvas with proper page size
        canvas_obj = canvas.Canvas(pdf_path, pagesize=(600, 800))
        
        # Add title with proper RTL alignment
        try:
            title_font = font_config.get("title_font", "Helvetica-Bold")
            canvas_obj.setFont(title_font, 18)
            
            # Always use Arabic title for consistency
            title_text = _format_text_unified("محتوى جلسة العلاج")  # "Therapy Session Content" in Arabic
            
            # Center the title
            canvas_obj.drawCentredString(300, 720, title_text)
            
        except Exception as e:
            print(f"⚠️ Error adding PDF title: {e}")
            # Fallback English title
            canvas_obj.setFont("Helvetica-Bold", 18)
            canvas_obj.drawCentredString(300, 720, "Therapy Session Content")
        
        # Add content with RTL alignment
        _draw_pdf_content(canvas_obj, content, font_config)
        
        # Save PDF
        canvas_obj.save()
        print(f"✅ PDF generated successfully with RTL alignment: {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        raise Exception(f"PDF generation failed: {str(e)}")

def _save_pdf_to_gridfs(pdf_path, patient_id, session_id):
    """Save PDF to GridFS with error handling"""
    try:
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            raise ValueError("PDF file is empty or doesn't exist")
        
        filename = _get_safe_filename(patient_id, session_id, ".pdf")
        
        with open(pdf_path, "rb") as f:
            pdf_file_id = fs.put(
                f,
                filename=filename,
                content_type="application/pdf"
            )
        
        print(f"✅ PDF saved to GridFS with ID: {pdf_file_id}")
        return str(pdf_file_id)
        
    except Exception as e:
        print(f"❌ Error saving PDF to GridFS: {e}")
        raise Exception(f"Failed to save PDF to GridFS: {str(e)}")

# ================================
# MAIN FUNCTION (Integrated approach)
# ================================

def speech_recognition_and_transcription(file_id, patient_id, session_id):
    """
    Main function: Process audio/video file for speech recognition and generate transcription PDF
    Integrates teammate's enhanced transcription logic with our architecture
    
    Args:
        file_id (str): GridFS file ID of the audio/video file
        patient_id (int): Patient ID
        session_id (int): Session ID
        
    Returns:
        str: GridFS file ID of the generated PDF
        
    Raises:
        Exception: If transcription process fails
    """
    print("🎤 Starting speech recognition and transcription...")
    print(f"   File ID: {file_id}")
    print(f"   Patient: {patient_id}, Session: {session_id}")
    
    # Input validation
    if not all([file_id, patient_id, session_id]):
        raise ValueError("Missing required parameters: file_id, patient_id, or session_id")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_files = []
        
        try:
            # Step 1: Download file from GridFS
            print("📥 Downloading file from GridFS...")
            input_file_path = os.path.join(tmpdir, "video.mp4")  # Teammate's naming
            
            try:
                get_file_from_gridfs(file_id, input_file_path)
            except Exception as e:
                raise Exception(f"Failed to download file from GridFS: {str(e)}")
            
            temp_files.append(input_file_path)
            print(f"✅ File downloaded: {input_file_path}")

            # Step 2: Process media file (teammate's approach)
            try:
                processed_audio_path = _process_media_file(input_file_path, tmpdir)
                temp_files.append(processed_audio_path)
            except Exception as e:
                raise Exception(f"Media processing failed: {str(e)}")
            
            # Step 3: Transcribe audio (teammate's method)
            try:
                raw_transcription = _transcribe_audio(processed_audio_path)
            except Exception as e:
                raise Exception(f"Audio transcription failed: {str(e)}")
            
            if not raw_transcription.strip():
                raise ValueError("Transcription resulted in empty content")
            
            # Step 4: Process with Gemini (teammate's enhanced prompt)
            try:
                refined_transcription = _process_with_gemini(raw_transcription)
            except Exception as e:
                print(f"⚠️ Gemini processing failed: {e}, using raw transcription")
                refined_transcription = raw_transcription
            
            # Step 5: Generate PDF (teammate's Arabic approach)
            try:
                pdf_path = os.path.join(tmpdir, _get_safe_filename(patient_id, session_id, ".pdf"))
                _generate_pdf(refined_transcription, pdf_path, patient_id, session_id)
                temp_files.append(pdf_path)
                print("✅ PDF generated with proper RTL alignment for Arabic content")
            except Exception as e:
                raise Exception(f"PDF generation failed: {str(e)}")
            
            # Step 6: Save PDF to GridFS
            try:
                pdf_file_id = _save_pdf_to_gridfs(pdf_path, patient_id, session_id)
            except Exception as e:
                raise Exception(f"Failed to save PDF to GridFS: {str(e)}")
            
            # Step 7: Update patient session with PDF reference
            try:
                update_transcription_pdf_reference(patient_id, session_id, pdf_file_id)
                print("✅ Patient session updated with PDF reference")
            except Exception as e:
                print(f"⚠️ Warning: Could not update patient session: {e}")
                # Don't fail the entire process for this
            
            print(f"✅ Speech recognition completed successfully!")
            print(f"   Generated PDF ID: {pdf_file_id}")
            return pdf_file_id
            
        except Exception as e:
            error_msg = f"❌ Error in speech recognition pipeline: {str(e)}"
            print(error_msg)
            raise Exception(f"Speech recognition failed: {str(e)}")
        finally:
            # Clean up temp files
            _cleanup_temp_files(*temp_files)

# ================================
# BACKWARD COMPATIBILITY
# ================================

def speech_recognition(file_id, patient_id, session_id):
    """Legacy function name - calls the new implementation"""
    return speech_recognition_and_transcription(file_id, patient_id, session_id)

# ================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# ================================

def get_supported_formats():
    """Get list of supported audio and video formats"""
    return {
        "video_formats": VIDEO_EXTENSIONS,
        "audio_formats": AUDIO_EXTENSIONS,
        "all_formats": VIDEO_EXTENSIONS + AUDIO_EXTENSIONS
    }

def validate_file_format(file_path):
    """Validate if file format is supported"""
    extension = os.path.splitext(file_path)[1].lower()
    supported = VIDEO_EXTENSIONS + AUDIO_EXTENSIONS
    return extension in supported

def get_service_status():
    """Get service status and configuration"""
    return {
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "whisper_model": "local" if WHISPER_MODEL_PATH else "downloaded",
        "whisper_model_path": WHISPER_MODEL_PATH,
        "gemini_available": GEMINI_MODEL is not None,
        "supported_formats": get_supported_formats(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "arabic_fonts_available": any(os.path.exists(p) for p in ARABIC_FONT_PATHS if p),
        "environment_configured": {
            "ffmpeg_path": bool(os.getenv("FFMPEG_PATH")),
            "gemini_api_key": bool(os.getenv("GEMINI_API_KEY")),
            "arabic_font_path": bool(os.getenv("ARABIC_FONT_PATH")),
            "whisper_model_path": bool(os.getenv("WHISPER_MODEL_PATH"))
        }
    }