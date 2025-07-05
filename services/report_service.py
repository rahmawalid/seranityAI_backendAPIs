"""
Complete Report Service Layer - TOV PDF Generation + Comprehensive FER+TOV
This combines both TOV-only and comprehensive (FER+TOV) report generation using ReportLab
"""

import os
import io
import pandas as pd
import google.generativeai as genai
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from PIL import Image
from bson import ObjectId
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import tempfile

from config import fs
from repository.patient_repository import (
    _get_patient_by_id, 
    _find_session_index,
    update_speech_report_pdf_reference
)
from repository.report_repository import (
    save_pdf_to_gridfs,
    get_file_from_gridfs,
    get_gridfs_file_object,
    save_gridout_images_to_tempfiles,
    load_dataframe_from_gridfs,
    validate_image_file
)
from model.patient_model import Patient, Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY", "AIzaSyBsDcl5tRJd6FR0fy0pNvwv76-S5QrVvK4"),
    "model_name": "gemini-2.0-flash",
    "temperature": 0.3
}

# Font paths (same as transcription service)
ARABIC_FONT_PATHS = [
    os.path.join(os.path.dirname(__file__), "Amiri-Regular.ttf"),
    os.path.abspath("Amiri-Regular.ttf"),
    "Amiri-Regular.ttf"
]

ARABIC_BOLD_FONT_PATHS = [
    os.path.join(os.path.dirname(__file__), "Amiri-Bold.ttf"),
    os.path.abspath("Amiri-Bold.ttf"),
    "Amiri-Bold.ttf"
]

LEFT_MARGIN = 50
RIGHT_MARGIN = 50
PAGE_WIDTH = 600  
TEXT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN


def remove_asterisks(text):
    """Remove asterisks while preserving structure"""
    return re.sub(r'\*+', '', text)

def _setup_fonts():
    """Setup fonts using the same logic as transcription service"""
    fonts_config = {
        "title_font": "Helvetica-Bold",
        "body_font": "Helvetica",
        "header_font": "Helvetica-Bold",
        "supports_arabic": False
    }
    
    logger.info("🔍 Looking for Amiri fonts...")
    
    # Try to register Amiri Regular
    amiri_regular_registered = False
    amiri_bold_registered = False
    
    for arabic_font_path in ARABIC_FONT_PATHS:
        if os.path.exists(arabic_font_path):
            try:
                pdfmetrics.registerFont(TTFont("Amiri", arabic_font_path))
                amiri_regular_registered = True
                logger.info(f"✅ Amiri Regular registered: {arabic_font_path}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Failed to register Amiri Regular {arabic_font_path}: {e}")
                continue
    
    # Try to register Amiri Bold
    for arabic_bold_path in ARABIC_BOLD_FONT_PATHS:
        if os.path.exists(arabic_bold_path):
            try:
                pdfmetrics.registerFont(TTFont("Amiri-Bold", arabic_bold_path))
                amiri_bold_registered = True
                logger.info(f"✅ Amiri Bold registered: {arabic_bold_path}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Failed to register Amiri Bold {arabic_bold_path}: {e}")
                continue
    
    # Configure fonts based on what was registered
    if amiri_regular_registered:
        if amiri_bold_registered:
            fonts_config.update({
                "title_font": "Amiri-Bold",
                "body_font": "Amiri", 
                "header_font": "Amiri-Bold",
                "supports_arabic": True
            })
            logger.info("✅ Using Amiri Bold + Regular with full RTL support")
        else:
            fonts_config.update({
                "title_font": "Amiri",
                "body_font": "Amiri", 
                "header_font": "Amiri",
                "supports_arabic": True
            })
            logger.info("✅ Using Amiri Regular with RTL support (no bold)")
    else:
        logger.info("❌ No Amiri fonts found, using Helvetica fallback")
    
    return fonts_config

def _format_text_unified(text):
    """Enhanced unified text formatting with better Arabic handling - SAME AS TRANSCRIPTION"""
    try:
        if not text:
            return ""
        
        text = str(text).strip()
        
        # Check if text contains Arabic characters - SAFE METHOD
        has_arabic = False
        try:
            for ch in text:
                if '\u0600' <= ch <= '\u06ff':
                    has_arabic = True
                    break
        except (TypeError, ValueError):
            has_arabic = False
        
        if has_arabic:
            # Apply Arabic text shaping and bidirectional algorithm
            try:
                # First reshape for proper Arabic glyph connection
                reshaped = reshape(text)
                # Then apply bidirectional algorithm for proper RTL layout
                bidi_text = get_display(reshaped)
                return bidi_text
            except Exception as e:
                logger.warning(f"⚠️ Arabic reshaping failed: {e}")
                return text
        else:
            # English text - return as-is but ensure it works with RTL layout
            return text
            
    except Exception as e:
        logger.warning(f"⚠️ Text formatting failed: {e}")
        return str(text) if text is not None else ""

def _add_pdf_header(canvas_obj, page_number, patient_id, session_id, font_config):
    """Add header to PDF - SAME LOGIC AS TRANSCRIPTION"""
    try:
        canvas_obj.setFont(font_config["header_font"], 14)
        
        header_text = f"Mental Health Analysis Report – Patient {patient_id} – Session {session_id}"
        page_text = f"Page {page_number}"
        
        # Format text
        formatted_header = _format_text_unified(header_text)
        formatted_page = _format_text_unified(page_text)
        
        if font_config["supports_arabic"]:
            # Use RTL layout for Arabic-capable fonts
            canvas_obj.drawRightString(550, 780, formatted_header)
            canvas_obj.drawString(LEFT_MARGIN, 780, formatted_page)
        else:
            # Fallback for non-Arabic fonts
            canvas_obj.drawString(LEFT_MARGIN, 780, formatted_header)
            canvas_obj.drawRightString(550, 780, formatted_page)
            
    except Exception as e:
        logger.warning(f"⚠️ Error adding PDF header: {e}")

def _wrap_text_for_pdf(text, font_config, max_width=400):
    """Wrap text for PDF - SAME AS TRANSCRIPTION"""
    try:
        if not text or not text.strip():
            return []
        
        # Format the text first
        formatted_text = _format_text_unified(text)
        
        # Simple word-based wrapping
        words = formatted_text.split()
        lines = []
        current_line = ""
        
        # Estimate characters per line
        chars_per_line = 60
        
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
        logger.warning(f"⚠️ Text wrapping error: {e}")
        return [_format_text_unified(text)]

def _draw_pdf_content_with_images(canvas_obj, content, font_config, fer_graph_paths=None):
    """Draw content with embedded images - FIXED VERSION FOR FER graphs"""
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
                y -= 5
                continue

            try:
                line = line.strip()

                # ✅ Detect and draw FER graph image if applicable
                if line.startswith("## FER Graph #") and fer_graph_paths:
                    try:
                        match = re.search(r"## FER Graph #(\d+)", line)
                        if not match:
                            continue
                        graph_number = int(match.group(1))
                        canvas_obj.setFont(font_config["header_font"], 13)
                        formatted_text = _format_text_unified(line.strip("## "))

                        if font_config["supports_arabic"]:
                            canvas_obj.drawRightString(550, y, formatted_text)
                        else:
                            canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                        y -= 25

                        # ✅ Validate index and draw image
                        if 0 <= graph_number - 1 < len(fer_graph_paths):
                            graph_path = fer_graph_paths[graph_number - 1]
                            if os.path.exists(graph_path):
                                logger.info(f"[✅] Embedding FER Graph #{graph_number} from path: {graph_path}")

                                # Ensure space is available
                                img_height = 300
                                if y < img_height + 50:
                                    canvas_obj.showPage()
                                    page_number += 1
                                    _add_pdf_header(canvas_obj, page_number,
                                                    font_config.get("patient_id", "UNKNOWN"),
                                                    font_config.get("session_id", "UNKNOWN"),
                                                    font_config)
                                    y = 670

                                # Embed image
                                canvas_obj.drawImage(graph_path, 50, y - img_height,
                                                     width=500, height=img_height,
                                                     preserveAspectRatio=True, mask='auto')
                                y -= img_height + 20
                            else:
                                logger.warning(f"[❌] Graph path not found: {graph_path}")
                                canvas_obj.setFont(font_config["body_font"], 11)
                                canvas_obj.drawString(LEFT_MARGIN, y, f"[Missing image: {graph_path}]")
                                y -= 20
                        else:
                            logger.warning(f"[⚠️] Graph number {graph_number} out of bounds for FER graph list.")
                        continue

                    except Exception as e:
                        logger.warning(f"[❌] Error embedding FER graph image: {e}")
                        continue

                # === HEADERS ===
                if line.startswith("###"):
                    canvas_obj.setFont(font_config["header_font"], 12)
                    formatted_text = _format_text_unified(line.strip("### "))
                    if font_config["supports_arabic"]:
                        canvas_obj.drawRightString(550, y, formatted_text)
                    else:
                        canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                    y -= 20

                elif line.startswith("##"):
                    canvas_obj.setFont(font_config["header_font"], 13)
                    formatted_text = _format_text_unified(line.strip("## "))
                    if font_config["supports_arabic"]:
                        canvas_obj.drawRightString(550, y, formatted_text)
                    else:
                        canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                    y -= 22

                elif line.startswith("#"):
                    canvas_obj.setFont(font_config["header_font"], 14)
                    formatted_text = _format_text_unified(line.strip("# "))
                    if font_config["supports_arabic"]:
                        canvas_obj.drawRightString(550, y, formatted_text)
                    else:
                        canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                    y -= 25

                # === BULLETS ===
                elif line.startswith("- ") or line.startswith("• "):
                    canvas_obj.setFont(font_config["body_font"], 11)
                    formatted_text = _format_text_unified(line)
                    if font_config["supports_arabic"]:
                        canvas_obj.drawRightString(530, y, formatted_text)
                    else:
                        canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                    y -= 18

                # === LABELS ===
                elif ":" in line and len(line) < 100:
                    canvas_obj.setFont(font_config["header_font"], 12)
                    formatted_text = _format_text_unified(line)
                    if font_config["supports_arabic"]:
                        canvas_obj.drawRightString(550, y, formatted_text)
                    else:
                        canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                    y -= 20

                # === PARAGRAPH ===
                else:
                    canvas_obj.setFont(font_config["body_font"], 11)
                    wrapped_lines = _wrap_text_for_pdf(line, font_config)
                    for wrapped_line in wrapped_lines:
                        if font_config["supports_arabic"]:
                            canvas_obj.drawRightString(550, y, wrapped_line)
                        else:
                            canvas_obj.drawString(LEFT_MARGIN, y, wrapped_line)
                        y -= 18
                y -= 5

                # Start new page if needed
                if y < 100:
                    canvas_obj.showPage()
                    page_number += 1
                    _add_pdf_header(canvas_obj, page_number,
                                    font_config.get("patient_id", "UNKNOWN"),
                                    font_config.get("session_id", "UNKNOWN"),
                                    font_config)
                    y = 670

            except Exception as e:
                logger.warning(f"[❌] Error processing line: {line[:50]} - {e}")
                continue

    except Exception as e:
        logger.error(f"❌ Error drawing PDF content with images: {e}")
        raise



def _generate_pdf_reportlab_with_images(content, pdf_path, patient_id, session_id, fer_graph_paths=None):
    """Generate PDF using ReportLab with embedded images - ENHANCED VERSION"""
    logger.info(f"📄 Generating PDF with embedded images using ReportLab: {pdf_path}")
    
    try:
        if not content or not content.strip():
            raise ValueError("Cannot generate PDF from empty content")
        
        # Setup fonts
        font_config = _setup_fonts()
        font_config["patient_id"] = patient_id
        font_config["session_id"] = session_id
        
        # Create PDF canvas with proper page size
        canvas_obj = canvas.Canvas(pdf_path, pagesize=(600, 800))
        
        # Add title
        try:
            title_font = font_config.get("title_font", "Helvetica-Bold")
            canvas_obj.setFont(title_font, 18)
            
            # Use English title for clinical reports
            title_text = _format_text_unified("Mental Health Analysis Report")
            
            # Center the title
            canvas_obj.drawCentredString(300, 720, title_text)
            
        except Exception as e:
            logger.warning(f"⚠️ Error adding PDF title: {e}")
            # Fallback title
            canvas_obj.setFont("Helvetica-Bold", 18)
            canvas_obj.drawCentredString(300, 720, "Mental Health Analysis Report")
        
        # Add content with images
        _draw_pdf_content_with_images(canvas_obj, content, font_config, fer_graph_paths)
        
        # Save PDF
        canvas_obj.save()
        logger.info(f"✅ PDF with embedded images generated successfully using ReportLab: {pdf_path}")
        
    except Exception as e:
        logger.error(f"❌ Error generating PDF with images: {e}")
        raise Exception(f"PDF generation with images failed: {str(e)}")

def _save_pdf_to_gridfs(pdf_path, patient_id, session_id):
    """Save PDF to GridFS - SAME AS TRANSCRIPTION"""
    try:
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            raise ValueError("PDF file is empty or doesn't exist")
        
        filename = f"mental_health_analysis_p{patient_id}_s{session_id}.pdf"
        
        with open(pdf_path, "rb") as f:
            pdf_file_id = fs.put(
                f,
                filename=filename,
                content_type="application/pdf"
            )
        
        logger.info(f"✅ PDF saved to GridFS with ID: {pdf_file_id}")
        return str(pdf_file_id)
        
    except Exception as e:
        logger.error(f"❌ Error saving PDF to GridFS: {e}")
        raise Exception(f"Failed to save PDF to GridFS: {str(e)}")

class ReportService:
    """Service class for handling patient report generation and management"""
    
    def __init__(self):
        """Initialize Gemini AI model"""
        try:
            genai.configure(api_key=GEMINI_CONFIG["api_key"])
            self.model = genai.GenerativeModel(GEMINI_CONFIG["model_name"])
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            raise ValueError(f"Failed to initialize Gemini AI: {str(e)}")
    
    def generate_session_analysis_report(self, patient_id: str, session_id: str) -> dict:
        """
        Generate comprehensive analysis report for a specific patient session
        """
        try:
            logger.info(f"Starting report generation for patient {patient_id}, session {session_id}")
            
            # Get patient and session data
            patient = _get_patient_by_id(patient_id)
            session_index = _find_session_index(patient, session_id)
            session = patient.sessions[session_index]
            
            # Extract feature data
            feature_files_data = self._extract_session_feature_files(session)
            
            # Validate required data
            self._validate_report_data(feature_files_data)
            
            # Generate the PDF report
            pdf_file_id = self._generate_comprehensive_report(
                session=session,
                patient=patient,
                fer_excel_id=feature_files_data["fer_excel_id"],
                tov_excel_id=feature_files_data["tov_excel_id"],
                fer_images=feature_files_data["fer_images"]
            )
            
            # Save PDF reference back to session
            update_speech_report_pdf_reference(patient_id, session_id, pdf_file_id)
            
            logger.info(f"✓ Generated analysis report for patient {patient_id}, session {session_id}")
            
            return {
                "report_id": pdf_file_id,
                "fer_excel": feature_files_data["fer_excel_file"],
                "fer_images": feature_files_data["fer_images"],
                "tov_excel": feature_files_data["tov_excel_file"],
                "session": session,
                "patient": patient,
                "text": session.text,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating session analysis report: {e}")
            raise ValueError(f"Failed to generate session analysis report: {str(e)}")
    
    def _extract_session_feature_files(self, session: Session) -> dict:
        """Extract and retrieve feature files from session data"""
        fer_data = session.feature_data.get("FER", {}) if session.feature_data else {}
        speech_data = session.feature_data.get("Speech", {}) if session.feature_data else {}
        
        # Get FER Excel file
        fer_excel_id = fer_data.get("fer_excel")
        fer_excel_file = None
        if fer_excel_id:
            try:
                fer_excel_file = get_gridfs_file_object(str(fer_excel_id))
            except Exception as e:
                logger.warning(f"Could not retrieve FER Excel file {fer_excel_id}: {e}")
        
        # Get FER plot images
        image_ids = fer_data.get("plot_images", [])
        fer_images = []
        for img_id in image_ids:
            try:
                fer_images.append(get_gridfs_file_object(str(img_id)))
            except Exception as e:
                logger.warning(f"Could not retrieve FER image {img_id}: {e}")
        
        # Get Speech/TOV Excel file
        tov_excel_id = speech_data.get("speech_excel")
        tov_excel_file = None
        if tov_excel_id:
            try:
                tov_excel_file = get_gridfs_file_object(str(tov_excel_id))
            except Exception as e:
                logger.warning(f"Could not retrieve TOV Excel file {tov_excel_id}: {e}")
        
        return {
            "fer_excel_id": fer_excel_id,
            "fer_excel_file": fer_excel_file,
            "fer_images": fer_images,
            "tov_excel_id": tov_excel_id,
            "tov_excel_file": tov_excel_file,
        }
    
    def _validate_report_data(self, feature_files_data: dict):
        """Validate that required data is available for report generation"""
        has_fer_data = feature_files_data["fer_excel_id"] is not None
        has_speech_data = feature_files_data["tov_excel_id"] is not None
        
        if not has_fer_data and not has_speech_data:
            raise ValueError("No analysis data available. Either FER or Speech data is required for report generation.")
        
        # Log available data for debugging
        logger.info("Report data validation:")
        logger.info(f"  - FER Excel: {'✓' if has_fer_data else '✗'}")
        logger.info(f"  - FER Images: {'✓' if feature_files_data['fer_images'] else '✗'}")
        logger.info(f"  - Speech Excel: {'✓' if has_speech_data else '✗'}")
    
    def _generate_comprehensive_report(self, session: Session, patient: Patient, 
                                     fer_excel_id: Optional[str], tov_excel_id: Optional[str], 
                                     fer_images: Optional[List] = None) -> str:
        """
        Generate comprehensive patient analysis report with AI analysis
        """
        try:
            logger.info(f"Generating comprehensive report for patient {patient.patientID}, session {session.session_id}")
            
            # Process FER images
            photo_paths = []
            if fer_images:
                try:
                    photo_paths = save_gridout_images_to_tempfiles(fer_images)
                    logger.info(f"Processed {len(photo_paths)} FER images")
                except Exception as e:
                    logger.warning(f"Failed to process FER images: {e}")
            
            # Determine analysis type and generate appropriate report
            has_fer = fer_excel_id is not None
            has_speech = tov_excel_id is not None
            
            if has_fer and has_speech:
                # Use comprehensive analysis
                return self._generate_fer_and_tov_analysis(session, patient, fer_excel_id, tov_excel_id, photo_paths)
            elif has_speech:
                # Use TOV-only analysis
                return self._generate_tov_only_analysis(session, patient, tov_excel_id)
            else:
                raise ValueError("No valid analysis data available")
                
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}")
            raise ValueError(f"Failed to generate comprehensive patient analysis report: {str(e)}")
    
    # ================================
    # TOV-ONLY ANALYSIS (WORKING)
    # ================================
    
    def _generate_tov_only_analysis(self, session: Session, patient: Patient, tov_excel_id: str) -> str:
        """
        Generate TOV-only analysis report - FIXED VERSION USING REPORTLAB
        """
        try:
            logger.info("Generating TOV-only analysis report...")
            
            # Extract session and patient data
            session_data = self._extract_session_info(session)
            patient_info = self._extract_patient_info(patient)
            
            # Load tone analysis data
            df_tone_analysis = load_dataframe_from_gridfs(tov_excel_id, file_type="excel")
            
            # Create tone summary
            tone_summary = self._create_tone_summary(df_tone_analysis)
            
            # Generate AI clinical analysis using TOV-only prompt
            ai_analysis_text = self._generate_ai_clinical_analysis_tov_only(patient_info, session_data, tone_summary)
            
            # Create complete content string (like transcription service)
            full_content = self._create_complete_content(patient_info, session_data, ai_analysis_text, df_tone_analysis)
            
            # Generate PDF using ReportLab (like transcription service) - NO IMAGES FOR TOV-ONLY
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, f"tov_analysis_p{patient_info.get('Patient ID', 'unknown')}_s{session_data.get('Session Number', 'unknown')}.pdf")
                
                _generate_pdf_reportlab(full_content, pdf_path, patient_info.get('Patient ID', 'unknown'), session_data.get('Session Number', 'unknown'))
                
                # Save to GridFS
                pdf_file_id = _save_pdf_to_gridfs(pdf_path, patient_info.get('Patient ID', 'unknown'), session_data.get('Session Number', 'unknown'))
            
            logger.info("✓ TOV-only PDF report created and saved to GridFS using ReportLab")
            logger.info("TOV-only analysis completed successfully!")
            return pdf_file_id
            
        except Exception as e:
            logger.error(f"TOV-only analysis failed: {e}")
            raise

    
    def _create_complete_content(self, patient_info: dict, session_data: dict, ai_analysis_text: str, df_tone_analysis: pd.DataFrame) -> str:
        """Create complete content string for PDF generation - like transcription service"""
        content_parts = []
        
        # Patient Information Section
        content_parts.append("# Patient Information")
        content_parts.append("")
        content_parts.append(f"Patient ID: {patient_info.get('Patient ID', 'Unknown')}")
        content_parts.append(f"Name: {patient_info.get('Full Name', 'Unknown')}")
        content_parts.append(f"Age: {patient_info.get('Age', 'Unknown')} | Gender: {patient_info.get('Gender', 'Unknown')}")
        content_parts.append(f"Occupation: {patient_info.get('Occupation', 'Unknown')}")
        content_parts.append(f"Marital Status: {patient_info.get('Marital Status', 'Unknown')}")
        content_parts.append(f"Primary Concern: {patient_info.get('Reason for Therapy', 'Not provided')}")
        content_parts.append("")
        
        # Session Information
        content_parts.append("# Session Information")
        content_parts.append("")
        content_parts.append(f"Session Number: {session_data.get('Session Number', 'Unknown')}")
        content_parts.append(f"Session Date: {session_data.get('Date of Session', 'Unknown')}")
        content_parts.append(f"Duration: {session_data.get('Duration', 'Unknown')} minutes")
        content_parts.append("")
        
        # Clinical Analysis
        content_parts.append("# Clinical Analysis")
        content_parts.append("")
        
        if ai_analysis_text and ai_analysis_text.strip():
            # Split analysis into lines and add them
            analysis_lines = ai_analysis_text.strip().split('\n')
            content_parts.extend(analysis_lines)
        else:
            content_parts.append("No clinical analysis available.")
        
        content_parts.append("")
        
        # Tone Analysis Summary
        content_parts.append("# Tone Analysis Summary")
        content_parts.append("")
        content_parts.append(f"Total Chunks Analyzed: {len(df_tone_analysis)}")
        content_parts.append("")
        
        # Add some sample chunks (limited to avoid very long content)
        if not df_tone_analysis.empty:
            content_parts.append("## Sample Analysis Chunks:")
            content_parts.append("")
            
            for idx, row in df_tone_analysis.head(10).iterrows():
                chunk_num = str(row.get('Chunk Number', 'Unknown'))
                prediction = str(row.get('Prediction', 'Unknown'))
                content_parts.append(f"Chunk {chunk_num}: {prediction}")
            
            if len(df_tone_analysis) > 10:
                content_parts.append(f"... and {len(df_tone_analysis) - 10} more chunks")
        
        return '\n'.join(content_parts)
    
    def _generate_ai_clinical_analysis_tov_only(self, patient_info: dict, session_data: dict, tone_summary: str) -> str:
        """Generate AI clinical analysis using TOV-only prompt"""
        try:
            logger.info("Generating TOV-only clinical analysis...")
            
            # Build TOV-only diagnostic prompt
            unified_prompt = f"""
**Role**: You are an advanced AI clinical psychologist analyzing patient-therapist sessions through text tone analysis.

=== BACKGROUND CLINICAL INFORMATION ===
• Primary Concern: {patient_info['Reason for Therapy']}
• Physical Health: {patient_info['Physical Health Conditions']}
• Family Mental Health History: {patient_info['Family History of Mental Illness']}
• Substance Use: {patient_info['Substance Use']}
• Current Medications: {patient_info['Current Medications']}

=== SESSION INFORMATION ===
• Session #: {session_data['Session Number']} | Date: {session_data['Date of Session']}
• Duration: {session_data['Duration']} mins | Type: {session_data['Session Type']}

=== TEXT TONE ANALYSIS ===
{tone_summary}

**Analysis Instructions**:

1. **Emotional Pattern Analysis**:
   - Identify dominant emotional patterns in text analysis
   - Analyze tone changes throughout the session
   - **EVIDENCE REQUIREMENT**: Quote exact transcriptions and cite specific data points

2. **Clinical Significance Evaluation**:
   - Evaluate each finding for clinical relevance
   - Highlight potential diagnostic indicators
   - Note any risk factors or red flags
   - **EVIDENCE REQUIREMENT**: Reference specific patient statements

3. **Diagnostic Insights**:
   - Suggest possible diagnoses with supporting evidence from DSM-5/ICD-11
   - Provide differential diagnosis considerations with confidence percentage
   - Rate confidence levels for each hypothesis

4. **Treatment Recommendations**:
   - Suggest therapeutic approaches based on findings
   - Recommend specific interventions
   - Provide session focus areas for next meeting
   - **EVIDENCE REQUIREMENT**: Base recommendations on specific documented behaviors

5. **Risk Assessment**:
   - Evaluate suicide risk if present
   - Assess danger to others if indicated
   - Note any urgent care needs
   - **EVIDENCE REQUIREMENT**: Quote exact statements related to risk factors

**CRITICAL EVIDENCE FORMATTING REQUIREMENTS**:
- Use **EVIDENCE:** tags before each quote
- Format quotes exactly as they appear in the data
- Include chunk numbers for transcript quotes

**Output Requirements**:
- Use DSM-5/ICD-11 terminology (this is mandatory)
- Structure findings by clinical priority
- Include direct evidence from the data
- Provide clear, actionable recommendations
- Maintain professional clinical tone
- START IMMEDIATELY with clinical analysis - no introductory phrases
- Begin directly with "# EMOTIONAL PATTERN ANALYSIS" or similar clinical heading

**CRITICAL INSTRUCTION:**
Begin your response IMMEDIATELY with the clinical analysis. Do not include any introductory phrases like "Here's a clinical analysis" or "Okay, here's". Start directly with a clinical heading such as "# EMOTIONAL PATTERN ANALYSIS" and proceed with the professional analysis.
"""
            
            generation_config = genai.types.GenerationConfig(temperature=0.3)
            response = self.model.generate_content(unified_prompt, generation_config=generation_config)
            
            clean_response_text = remove_asterisks(response.text)
            
            logger.info("✓ TOV-only AI clinical analysis generated successfully")
            return clean_response_text
            
        except Exception as e:
            logger.error(f"Error generating TOV-only AI analysis: {e}")
            raise
    
    # ================================
    # COMPREHENSIVE FER+TOV ANALYSIS (FIXED)
    # ================================
    
    def _generate_fer_and_tov_analysis(self, session: Session, patient: Patient, fer_excel_id: str, 
                                     tov_excel_id: str, photo_paths: List[str]) -> str:
        """
        Generate FER+TOV comprehensive analysis report using ReportLab - FIXED VERSION
        """
        try:
            logger.info("Generating FER+TOV comprehensive analysis report...")
            
            # Extract session and patient data
            session_data = self._extract_session_info(session)
            patient_info = self._extract_patient_info(patient)
            
            # Set analysis parameters
            fps = 30
            TEXT_CHUNK_DURATION_SECONDS = 15
            FER_ROWS_PER_TEXT_CHUNK = int(fps * TEXT_CHUNK_DURATION_SECONDS)
            
            # Load data files
            text_tones_df = load_dataframe_from_gridfs(tov_excel_id, file_type="excel")
            fer_df = load_dataframe_from_gridfs(fer_excel_id, file_type="excel")
            
            # FER analysis
            fer_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            fer_df['Dominant_FER'] = fer_df[fer_emotions].idxmax(axis=1)
            fer_df['chunk_number'] = (fer_df.index // FER_ROWS_PER_TEXT_CHUNK) + 1
            
            # Get dominant FER per chunk
            dominant_fer = fer_df.groupby('chunk_number')['Dominant_FER'].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'neutral'
            ).reset_index()
            dominant_fer.columns = ['Chunk Number', 'Dominant_FER']
            
            # Get confidence scores
            confidence_scores = fer_df.groupby('chunk_number', group_keys=False).apply(
                self._get_mean_confidence
            ).reset_index()
            confidence_scores.columns = ['Chunk Number', 'FER_Confidence']
            
            # Merge analysis
            merged_analysis = pd.merge(
                text_tones_df, dominant_fer, on='Chunk Number', how='left'
            ).merge(confidence_scores, on='Chunk Number', how='left')
            
            # Calculate mismatches
            merged_analysis['Mismatch'] = (
                merged_analysis['Prediction'].str.lower() != 
                merged_analysis['Dominant_FER'].str.lower()
            )
            
            mismatch_percentage = (merged_analysis['Mismatch'].sum() / len(merged_analysis)) * 100
            critical_segments = merged_analysis[merged_analysis['Mismatch']].nlargest(3, 'FER_Confidence')
            
            # Generate insights
            fer_insights = self._generate_fer_insights(merged_analysis)
            
            # Create combined text analysis
            text_analysis = self._create_combined_text_analysis(merged_analysis)
            
            # Analyze FER graphs individually
            fer_graph_descriptions = self._analyze_individual_fer_graphs(photo_paths)
            
            # Generate AI clinical analysis using comprehensive prompt
            ai_analysis_text = self._generate_ai_clinical_analysis_comprehensive(
                patient_info, session_data, text_analysis, fer_graph_descriptions,
                fer_insights, mismatch_percentage, critical_segments
            )
            
            # Create comprehensive content string (like TOV-only approach)
            full_content = self._create_comprehensive_content(
                patient_info, session_data, ai_analysis_text, merged_analysis,
                fer_graph_descriptions, mismatch_percentage
            )
            
            # Generate PDF using ReportLab with embedded images - ENHANCED VERSION
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, f"comprehensive_analysis_p{patient_info.get('Patient ID', 'unknown')}_s{session_data.get('Session Number', 'unknown')}.pdf")
                
                # Use the enhanced PDF generation function that embeds images
                _generate_pdf_reportlab_with_images(full_content, pdf_path, 
                                                  patient_info.get('Patient ID', 'unknown'), 
                                                  session_data.get('Session Number', 'unknown'),
                                                  photo_paths)  # Pass the actual graph file paths
                
                # Save to GridFS
                pdf_file_id = _save_pdf_to_gridfs(pdf_path, patient_info.get('Patient ID', 'unknown'), session_data.get('Session Number', 'unknown'))
            
            logger.info("✓ Comprehensive FER+TOV PDF report created and saved to GridFS using ReportLab")
            logger.info("FER+TOV comprehensive analysis completed successfully!")
            return pdf_file_id
            
        except Exception as e:
            logger.error(f"FER+TOV comprehensive analysis failed: {e}")
            raise

    def _get_mean_confidence(self, chunk_group: pd.DataFrame) -> float:
        """Calculate mean confidence for dominant emotion in chunk"""
        try:
            if chunk_group.empty:
                return 0.0
            dominant = chunk_group['Dominant_FER'].mode()[0] if not chunk_group['Dominant_FER'].mode().empty else 'neutral'
            if dominant not in chunk_group.columns:
                return 0.0
            confidence_values = chunk_group.loc[chunk_group['Dominant_FER'] == dominant, dominant]
            return confidence_values.mean() if not confidence_values.empty else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate mean confidence: {e}")
            return 0.0
    
    def _generate_fer_insights(self, analysis_df: pd.DataFrame) -> str:
        """Generate FER insights from analysis DataFrame"""
        try:
            if analysis_df.empty or 'Mismatch' not in analysis_df.columns:
                return "No FER insights available due to insufficient data."
            
            frequent_mismatches = analysis_df[analysis_df['Mismatch']].groupby(
                ['Prediction', 'Dominant_FER']
            ).size().nlargest(3)
            
            insights = []
            for (text_tone, fer_exp), count in frequent_mismatches.items():
                interpretation = {
                    ('angry', 'neutral'): "Possible emotional suppression",
                    ('angry', 'happy'): "Potential masking of true feelings",
                    ('sad', 'neutral'): "Flat affect may indicate depression",
                }.get((text_tone.lower(), fer_exp.lower()), "Requires clinical evaluation")
                
                insights.append(
                    f"- {text_tone} speech with {fer_exp} face ({count}x): {interpretation}"
                )
            return '\n'.join(insights)
            
        except Exception as e:
            logger.error(f"Failed to generate FER insights: {e}")
            return "FER insights unavailable due to processing error."
    
    def _analyze_individual_fer_graphs(self, graph_paths: List[str]) -> dict:
        """Analyze individual FER graphs using AI - FIXED VERSION"""
        fer_graph_descriptions = {}
        
        for i, path in enumerate(graph_paths):
            graph_number = i + 1
            individual_graph_prompt = f"""
            **FER GRAPH #{graph_number} ANALYSIS**

            Analyze this facial expression recognition graph and provide a comprehensive description using the EXACT format below:

            **MANDATORY FORMAT - Follow this structure exactly:**

            1. **Emotion Types**: [List all emotions present: angry, disgust, fear, happy, neutral, sad, surprise]

            2. **Dominant Emotions**: [Which emotions appear most frequently or have highest intensity levels, with specific percentages/values if visible]

            3. **Temporal Patterns**: [Describe how emotions change over the time period shown, including specific time ranges and patterns]

            4. **Intensity Levels**: [Describe the strength/confidence levels of detected emotions with specific numerical values where visible]

            5. **Overall Emotional Journey**: [Summarize the emotional progression from start to end of the time period]

            6. **Notable Peaks/Valleys**: [Highlight specific time points where emotions spike or drop significantly, with timestamps/coordinates]

            7. **Emotional Stability**: [Assess whether the emotional state is consistent, variable, or highly fluctuating throughout the period]

            **CRITICAL INSTRUCTIONS:**
            - Use EXACTLY this numbered format (1. through 7.)
            - Start immediately with "1. **Emotion Types**:" - no introductory text
            - Include specific numerical values, timestamps, and intensity levels when visible
            - Be precise about time ranges and emotional transitions
            - Maintain consistent formatting across all sections
            """
            
            try:
                if os.path.exists(path):
                    img = Image.open(path)
                    res = self.model.generate_content([individual_graph_prompt, img])
                    clean_description = remove_asterisks(res.text)
                    fer_graph_descriptions[graph_number] = {
                        'path': path,
                        'description': clean_description,
                        'success': True,
                        'filename': os.path.basename(path)
                    }
                    logger.info(f"✓ Successfully analyzed FER Graph #{graph_number}: {os.path.basename(path)}")
                else:
                    fer_graph_descriptions[graph_number] = {
                        'path': path,
                        'description': f"Graph file not found: {path}",
                        'success': False,
                        'filename': os.path.basename(path)
                    }
                    logger.warning(f"✗ FER Graph #{graph_number} not found: {path}")
                    
            except Exception as e:
                fer_graph_descriptions[graph_number] = {
                    'path': path,
                    'description': f"Error analyzing graph: {str(e)}",
                    'success': False,
                    'filename': os.path.basename(path)
                }
                logger.error(f"Error analyzing FER Graph #{graph_number} ({path}): {str(e)}")
        
        return fer_graph_descriptions
    
    def _create_combined_text_analysis(self, merged_analysis: pd.DataFrame) -> str:
        """Create combined text analysis from merged DataFrame"""
        combined_data = []
        for _, row in merged_analysis.iterrows():
            combined_data.append(
                f"Chunk {row['Chunk Number']}: "
                f"\"{row['Transcription']}\" | "
                f"Text Tone: {row['Prediction']} | "
                f"Facial Expression: {row['Dominant_FER']}"
            )
        return "\n".join(combined_data)
    
    def _generate_ai_clinical_analysis_comprehensive(self, patient_info: dict, session_data: dict, 
                                                   text_analysis: str, fer_graph_descriptions: dict,
                                                   fer_insights: str, mismatch_percentage: float, 
                                                   critical_segments: pd.DataFrame) -> str:
        """Generate AI clinical analysis using comprehensive prompt"""
        try:
            logger.info("Generating comprehensive clinical analysis...")
            
            # Build individual graph descriptions section
            fer_graphs_section = "\n".join([
        f"=== FER GRAPH #{num} ({data['filename']}) ===\n{data['description']}\n"
        for num, data in fer_graph_descriptions.items() if data['success']
    ])
            
            # Build comprehensive diagnostic prompt
            unified_prompt =  f"""

*Role*: You are an advanced AI clinical psychologist
analyzing patient-therapist sessions through
combined text tone and facial expression analysis .

=== BACKGROUND CLINICAL INFORMATION ===
* Primary Concern: {patient_info['Reason for Therapy']}
* Physical Health: {patient_info['Physical Health Conditions']}
* Family Mental Health History: {patient_info['Family History of Mental Illness']}
* Substance Use: {patient_info['Substance Use']}
* Current Medications: {patient_info['Current Medications']}

=== SESSION INFORMATION ===
* Session #: {session_data['Session Number']} | Date: {session_data['Date of Session']}
* Duration: {session_data['Duration']} mins | Type: {session_data['Session Type']}




=== VIDEO/AUDIO TRANSCRIPT ANALYSIS ===
* Overall Mismatch Rate: {mismatch_percentage:.1f}%
* Critical Segments: {len(critical_segments)} high-confidence mismatches detected

Detailed Transcript Analysis:
{text_analysis}

*Analysis Instructions*:

1. *Emotional Pattern Analysis*:
   - Identify dominant emotional patterns in text and facial expressions
   - Analyze alignment/misalignment between verbal and non-verbal cues
   - Detect emotional suppression or amplification patterns
   - *EVIDENCE REQUIREMENT*: Quote exact transcriptions and cite specific data points

2. *Clinical Significance Evaluation*:
   - Evaluate each finding for clinical relevance
   - Highlight potential diagnostic indicators
   - Note any risk factors or red flags
   - *EVIDENCE REQUIREMENT*: Reference specific therapist observations and patient statements

3. *Diagnostic Insights*:
   - Suggest possible diagnoses with supporting evidence
   - Provide differential diagnosis considerations with percentage
   - Rate confidence levels for each hypothesis


4. *Treatment Recommendations*:
   - Suggest therapeutic approaches
   - Recommend specific interventions
   - Provide session focus areas for next meeting
   - *EVIDENCE REQUIREMENT*: Base recommendations on specific documented behaviors

5. *Risk Assessment*:
   - Evaluate suicide risk if present
   - Assess danger to others if indicated
   - Note any urgent care needs
   - *EVIDENCE REQUIREMENT*: Quote exact statements related to risk factors

*CRITICAL EVIDENCE FORMATTING REQUIREMENTS*:
- Use *EVIDENCE:* tags before each quote
- Format quotes exactly as they appear in the data
- Include chunk numbers for transcript quotes
- Reference specific therapist note sections

*Output Requirements*:
- Use DSM-5/ICD-11 terminology it is a must
- Structure findings by clinical priority
- Include direct evidence from the data
- Provide clear, actionable recommendations
- Maintain professional clinical tone
- START IMMEDIATELY with clinical analysis - no introductory phrases
- Begin directly with "# EMOTIONAL PATTERN ANALYSIS" or similar clinical heading

"""
            
            generation_config = genai.types.GenerationConfig(temperature=0.3)
            response = self.model.generate_content(unified_prompt, generation_config=generation_config)
            
            clean_response_text = remove_asterisks(response.text)
            
            logger.info("✓ Comprehensive AI clinical analysis generated successfully")
            return clean_response_text
            
        except Exception as e:
            logger.error(f"Error generating comprehensive AI analysis: {e}")
            raise
    
    def _create_comprehensive_content(self, patient_info: dict, session_data: dict,
                                    ai_analysis_text: str, merged_analysis: pd.DataFrame,
                                    fer_graph_descriptions: dict, mismatch_percentage: float) -> str:
        """Create comprehensive content string for PDF generation - SAME PATTERN AS TOV-ONLY"""
        content_parts = []
        
        # Patient Information Section
        content_parts.append("# Patient Information")
        content_parts.append("")
        content_parts.append(f"Patient ID: {patient_info.get('Patient ID', 'Unknown')}")
        content_parts.append(f"Name: {patient_info.get('Full Name', 'Unknown')}")
        content_parts.append(f"Age: {patient_info.get('Age', 'Unknown')} | Gender: {patient_info.get('Gender', 'Unknown')}")
        content_parts.append(f"Occupation: {patient_info.get('Occupation', 'Unknown')}")
        content_parts.append(f"Marital Status: {patient_info.get('Marital Status', 'Unknown')}")
        content_parts.append(f"Primary Concern: {patient_info.get('Reason for Therapy', 'Not provided')}")
        content_parts.append("")
        
        # Session Information
        content_parts.append("# Session Information")
        content_parts.append("")
        content_parts.append(f"Session Number: {session_data.get('Session Number', 'Unknown')}")
        content_parts.append(f"Session Date: {session_data.get('Date of Session', 'Unknown')}")
        content_parts.append(f"Duration: {session_data.get('Duration', 'Unknown')} minutes")
        content_parts.append("")
        
        # Analysis Statistics
        content_parts.append("# Analysis Statistics")
        content_parts.append("")
        mismatch_count = merged_analysis['Mismatch'].sum() if 'Mismatch' in merged_analysis.columns else 0
        content_parts.append(f"Overall Mismatch Rate: {mismatch_percentage:.1f}%")
        content_parts.append(f"Total Chunks Analyzed: {len(merged_analysis)}")
        content_parts.append(f"FER Images Processed: {len(fer_graph_descriptions)}")
        content_parts.append(f"Critical Mismatches: {mismatch_count}")
        content_parts.append("")
        
        # Clinical Analysis
        content_parts.append("# Clinical Analysis")
        content_parts.append("")
        
        if ai_analysis_text and ai_analysis_text.strip():
            # Split analysis into lines and add them
            analysis_lines = ai_analysis_text.strip().split('\n')
            content_parts.extend(analysis_lines)
        else:
            content_parts.append("No clinical analysis available.")
        
        content_parts.append("")
        
        # FER Graph Analysis
        if fer_graph_descriptions:
            content_parts.append("# FER Graph Analysis")
            content_parts.append("")
            
            for graph_number, graph_data in fer_graph_descriptions.items():
                if graph_data.get('success', False):
                    content_parts.append(f"## FER Graph #{graph_number}: {graph_data.get('filename', 'Unknown')}")
                    content_parts.append("")
                    
                    description = graph_data.get('description', 'No description available')
                    description_lines = description.split('\n')
                    content_parts.extend(description_lines)
                    content_parts.append("")
                else:
                    content_parts.append(f"## FER Graph #{graph_number}: Error")
                    content_parts.append(f"Could not analyze: {graph_data.get('description', 'Unknown error')}")
                    content_parts.append("")
        
        # Mismatch Analysis Summary
        if not merged_analysis.empty and 'Mismatch' in merged_analysis.columns:
            content_parts.append("# Mismatch Analysis Summary")
            content_parts.append("")
            content_parts.append(f"Total Mismatches: {mismatch_count} out of {len(merged_analysis)} chunks")
            content_parts.append("")
            
            # Add some sample mismatches
            mismatches = merged_analysis[merged_analysis['Mismatch']]
            if not mismatches.empty:
                content_parts.append("## Key Mismatches:")
                content_parts.append("")
                
                for idx, row in mismatches.head(5).iterrows():
                    chunk_num = str(row.get('Chunk Number', 'Unknown'))
                    text_tone = str(row.get('Prediction', 'Unknown'))
                    fer_emotion = str(row.get('Dominant_FER', 'Unknown'))
                    transcription = str(row.get('Transcription', 'No text'))[:100] + "..."
                    content_parts.append(f"Chunk {chunk_num}: \"{transcription}\"")
                    content_parts.append(f"  Text Tone: {text_tone} | Facial Expression: {fer_emotion}")
                    content_parts.append("")
                
                if len(mismatches) > 5:
                    content_parts.append(f"... and {len(mismatches) - 5} more mismatches")
        
        return '\n'.join(content_parts)
    
    # ================================
    # SHARED HELPER METHODS
    # ================================
    
    def _extract_session_info(self, session: Session) -> dict:
        """Extract session information for prompts"""
        return {
            'Session Number': getattr(session, 'session_id', 'Unknown'),
            'Date of Session': getattr(session, 'date', 'Unknown'),
            'Duration': getattr(session, 'duration', 'Unknown'),
            'Session Type': getattr(session, 'session_type', 'Individual')
        }
    
    def _extract_patient_info(self, patient: Patient) -> dict:
        """Extract patient information for prompts"""
        info = {
            'Patient ID': getattr(patient, 'patientID', 'Unknown'),
            'Full Name': 'Unknown',
            'Age': 'Unknown',
            'Gender': 'Unknown',
            'Occupation': 'Unknown',
            'Marital Status': 'Unknown',
            'Reason for Therapy': 'Not provided',
            'Physical Health Conditions': 'Not provided',
            'Family History of Mental Illness': 'Not provided',
            'Substance Use': 'Not provided',
            'Current Medications': 'Not provided'
        }
        
        if patient.personalInfo:
            info['Full Name'] = getattr(patient.personalInfo, 'full_name', 'Unknown')
            info['Age'] = getattr(patient.personalInfo, 'age', 'Unknown')
            info['Gender'] = getattr(patient.personalInfo, 'gender', 'Unknown')
            info['Occupation'] = getattr(patient.personalInfo, 'occupation', 'Unknown')
            info['Marital Status'] = getattr(patient.personalInfo, 'marital_status', 'Unknown')
            
            if hasattr(patient.personalInfo, 'therapy_info') and patient.personalInfo.therapy_info:
                info['Reason for Therapy'] = getattr(patient.personalInfo.therapy_info, 'reason_for_therapy', 'Not provided')
            
            if hasattr(patient.personalInfo, 'health_info') and patient.personalInfo.health_info:
                health_info = patient.personalInfo.health_info
                info['Physical Health Conditions'] = getattr(health_info, 'physical_health_conditions', 'Not provided')
                info['Family History of Mental Illness'] = getattr(health_info, 'family_history_of_mental_illness', 'Not provided')
                info['Substance Use'] = getattr(health_info, 'substance_use', 'Not provided')
                info['Current Medications'] = getattr(health_info, 'current_medications', 'Not provided')
        
        return info
    
    def _create_tone_summary(self, df_tone_analysis: pd.DataFrame) -> str:
        """Create tone summary from DataFrame"""
        tone_summary = "\n".join([
            f"Chunk {row['Chunk Number']}: \"{row['Transcription']}\" - Tone: {row['Prediction']}"
            for _, row in df_tone_analysis.iterrows()
        ])
        return tone_summary
    
    # ================================
    # METADATA AND UTILITY METHODS
    # ================================
    
    def get_report_metadata(self, patient_id: str, session_id: str) -> dict:
        """Get metadata about available report data without generating the report"""
        try:
            patient = _get_patient_by_id(patient_id)
            session_index = _find_session_index(patient, session_id)
            session = patient.sessions[session_index]
            
            feature_files_data = self._extract_session_feature_files(session)
            
            return {
                "patient_id": patient_id,
                "session_id": session_id,
                "has_existing_report": session.report is not None,
                "existing_report_id": str(session.report) if session.report else None,
                "available_data": {
                    "fer_excel": feature_files_data["fer_excel_id"] is not None,
                    "fer_images": len(feature_files_data["fer_images"]) > 0,
                    "speech_excel": feature_files_data["tov_excel_id"] is not None,
                    "session_text": session.text is not None and len(session.text.strip()) > 0
                },
                "can_generate_report": (
                    feature_files_data["fer_excel_id"] is not None or 
                    feature_files_data["tov_excel_id"] is not None
                )
            }
            
        except Exception as e:
            raise ValueError(f"Failed to get report metadata: {str(e)}")
    
    def download_session_report(self, patient_id: str, session_id: str) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
        """Download existing report for a session, or generate if it doesn't exist"""
        try:
            patient = _get_patient_by_id(patient_id)
            session_index = _find_session_index(patient, session_id)
            session = patient.sessions[session_index]
            
            # Check if report already exists
            if session.report:
                try:
                    return get_file_from_gridfs(str(session.report))
                except Exception as e:
                    logger.warning(f"Could not retrieve existing report {session.report}: {e}")
            
            # Generate new report if none exists
            logger.info(f"No existing report found. Generating new report for patient {patient_id}, session {session_id}")
            report_data = self.generate_session_analysis_report(patient_id, session_id)
            
            # Download the newly generated report
            if report_data["report_id"]:
                return get_file_from_gridfs(report_data["report_id"])
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"❌ Error downloading session report: {e}")
            return None, None, None
        """Download existing report for a session, or generate if it doesn't exist"""
        try:
            patient = _get_patient_by_id(patient_id)
            session_index = _find_session_index(patient, session_id)
            session = patient.sessions[session_index]
            
            # Check if report already exists
            if session.report:
                try:
                    return get_file_from_gridfs(str(session.report))
                except Exception as e:
                    logger.warning(f"Could not retrieve existing report {session.report}: {e}")
            
            # Generate new report if none exists
            logger.info(f"No existing report found. Generating new report for patient {patient_id}, session {session_id}")
            report_data = self.generate_session_analysis_report(patient_id, session_id)
            
            # Download the newly generated report
            if report_data["report_id"]:
                return get_file_from_gridfs(report_data["report_id"])
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"❌ Error downloading session report: {e}")
            return None, None, None
        

def _generate_pdf_reportlab(content, pdf_path, patient_id, session_id):
        """ Generate PDF using ReportLab - ORIGINAL VERSION FOR TOV-ONLY"""
        logger.info(f"📄 Generating PDF using ReportLab: {pdf_path}")
        
        try:
            if not content or not content.strip():
                raise ValueError("Cannot generate PDF from empty content")
            
            # Setup fonts
            font_config = _setup_fonts()
            font_config["patient_id"] = patient_id
            font_config["session_id"] = session_id
            
            # Create PDF canvas with proper page size
            canvas_obj = canvas.Canvas(pdf_path, pagesize=(600, 800))
            
            # Add title
            try:
                title_font = font_config.get("title_font", "Helvetica-Bold")
                canvas_obj.setFont(title_font, 18)
                
                # Use English title for clinical reports
                title_text = _format_text_unified("Mental Health Analysis Report")
                
                # Center the title
                canvas_obj.drawCentredString(300, 720, title_text)
                
            except Exception as e:
                logger.warning(f"⚠️ Error adding PDF title: {e}")
                # Fallback title
                canvas_obj.setFont("Helvetica-Bold", 18)
                canvas_obj.drawCentredString(300, 720, "Mental Health Analysis Report")
            
            # Add content (original method - no images)
            _draw_pdf_content(canvas_obj, content, font_config)
            
            # Save PDF
            canvas_obj.save()
            logger.info(f"✅ PDF generated successfully using ReportLab: {pdf_path}")
            
        except Exception as e:
            logger.error(f"❌ Error generating PDF: {e}")
            raise Exception(f"PDF generation failed: {str(e)}")

def _draw_pdf_content(canvas_obj, content, font_config):
        """Draw content - ORIGINAL VERSION FOR TOV-ONLY (NO IMAGES)"""
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
                    # Process different line types (similar to transcription but for clinical content)
                    line = line.strip()
                    
                    if line.startswith("###"):
                        # Sub-subsection
                        canvas_obj.setFont(font_config["header_font"], 12)
                        formatted_text = _format_text_unified(line.strip("### "))
                        
                        if font_config["supports_arabic"]:
                            canvas_obj.drawRightString(550, y, formatted_text)
                        else:
                            canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                        y -= 20
                        
                    elif line.startswith("##"):
                        # Subsection
                        canvas_obj.setFont(font_config["header_font"], 13)
                        formatted_text = _format_text_unified(line.strip("## "))
                        
                        if font_config["supports_arabic"]:
                            canvas_obj.drawRightString(550, y, formatted_text)
                        else:
                            canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                        y -= 22
                        
                    elif line.startswith("#"):
                        # Main section
                        canvas_obj.setFont(font_config["header_font"], 14)
                        formatted_text = _format_text_unified(line.strip("# "))
                        
                        if font_config["supports_arabic"]:
                            canvas_obj.drawRightString(550, y, formatted_text)
                        else:
                            canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                        y -= 25
                        
                    elif line.startswith("- ") or line.startswith("• "):
                        # Bullet points
                        canvas_obj.setFont(font_config["body_font"], 11)
                        formatted_text = _format_text_unified(line)
                        
                        if font_config["supports_arabic"]:
                            canvas_obj.drawRightString(530, y, formatted_text)  # Slightly indented
                        else:
                            canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)  # Indented from left
                        y -= 18
                        
                    elif ":" in line and len(line) < 100:
                        # Labels/headers
                        canvas_obj.setFont(font_config["header_font"], 12)
                        formatted_text = _format_text_unified(line)
                        
                        if font_config["supports_arabic"]:
                            canvas_obj.drawRightString(550, y, formatted_text)
                        else:
                            canvas_obj.drawString(LEFT_MARGIN, y, formatted_text)
                        y -= 20
                        
                    else:
                        # Regular content - wrap if needed
                        canvas_obj.setFont(font_config["body_font"], 11)
                        wrapped_lines = _wrap_text_for_pdf(line, font_config)
                        
                        for wrapped_line in wrapped_lines:
                            if font_config["supports_arabic"]:
                                canvas_obj.drawRightString(550, y, wrapped_line)
                            else:
                                canvas_obj.drawString(LEFT_MARGIN, y, wrapped_line)
                            y -= 18
                    
                    y -= 5  # Extra spacing between blocks
                    
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
                    logger.warning(f"⚠️ Error drawing line '{line[:50]}...': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error drawing PDF content: {e}")
            raise
    