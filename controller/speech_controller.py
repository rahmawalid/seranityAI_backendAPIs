"""
Speech Controller - HTTP Request Handler Layer
Handles all HTTP requests for speech analysis and tone-of-voice functionality
Enhanced with better error handling, CORS support, and alignment with architecture
"""

import os
import tempfile
import logging
import traceback
from io import BytesIO
from flask import Blueprint, request, jsonify, send_file, Response, make_response
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from bson import ObjectId
import gridfs
from pymongo import MongoClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Service Layer Imports (Controllers should use services)
from services.TOV_service import process_speech_and_tov_by_id

# Repository Layer Imports
from repository.file_repository import save_file_to_gridfs
from repository.patient_repository import (
    get_patient_by_id,
    attach_video_to_session,
    download_report_by_id
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Blueprint setup
speech_bp = Blueprint('speech', __name__)

# MongoDB/GridFS setup
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

# File configuration
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav', 'flac'}
MAX_FILE_SIZE_MB = 500  # 500MB max file size

# ================================
# HELPER FUNCTIONS
# ================================

def _cors_preflight():
    """Handle CORS preflight requests"""
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

def _handle_controller_error(error, default_status=500):
    """Consistent error handling for controller endpoints"""
    logger.error(f"Speech controller error: {error}")
    
    if isinstance(error, ValueError):
        return jsonify({"error": "Invalid input", "details": str(error)}), 400
    elif isinstance(error, FileNotFoundError) or "not found" in str(error).lower():
        return jsonify({"error": "Resource not found", "details": str(error)}), 404
    elif "gridfs" in str(error).lower() or "NoFile" in str(error):
        return jsonify({"error": "File not found", "details": str(error)}), 404
    else:
        return jsonify({"error": "Internal server error", "details": str(error)}), default_status

def allowed_file(filename):
    """Check if file extension is allowed - Original logic preserved"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _validate_file_upload(file, max_size_mb=MAX_FILE_SIZE_MB):
    """Validate file upload with proper error messages"""
    if not file or file.filename == "":
        raise ValueError("No file provided")
    
    filename = secure_filename(file.filename)
    if not filename:
        raise ValueError("Invalid filename")
    
    # Check extension using original logic
    if not allowed_file(filename):
        raise ValueError(f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    
    # Check file size if possible
    try:
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning
        
        max_size_bytes = max_size_mb * 1024 * 1024
        if size > max_size_bytes:
            raise ValueError(f"File too large. Maximum size: {max_size_mb}MB")
    except Exception:
        # Skip size check if not supported
        pass
    
    return filename

def _validate_patient_and_session(patient_id, session_id):
    """Validate patient and session existence"""
    try:
        patient = get_patient_by_id(patient_id)
        session_found = any(sess.session_id == session_id for sess in patient.sessions)
        if not session_found:
            raise ValueError(f"Session {session_id} not found for patient {patient_id}")
        return patient
    except Exception as e:
        if "not found" in str(e).lower():
            raise ValueError(f"Patient {patient_id} not found")
        raise e

def _validate_file_id_and_exists(file_id):
    """Validate file ID format and existence in GridFS"""
    if not ObjectId.is_valid(file_id):
        raise ValueError("Invalid file ID format")
    
    try:
        fs.get(ObjectId(file_id))
    except gridfs.errors.NoFile:
        raise ValueError("File not found in storage")

# ================================
# SPEECH ANALYSIS ENDPOINTS (ORIGINAL LOGIC PRESERVED)
# ================================

@speech_bp.route('/upload-video/<int:patient_id>/<int:session_id>', methods=['POST', 'OPTIONS'])
@cross_origin()
def upload_video_for_speech(patient_id, session_id):
    """
    Upload a video file for speech analysis - Enhanced with CORS and error handling
    Original logic preserved exactly
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"📤 Starting video upload for patient {patient_id}, session {session_id}")
        
        # Original validation logic preserved
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: mp4, avi, mov, mkv, webm, mp3, wav, flac"}), 400

        # Verify patient and session exist - Original logic preserved
        try:
            patient = get_patient_by_id(patient_id)
            session_found = any(sess.session_id == session_id for sess in patient.sessions)
            if not session_found:
                return jsonify({"error": f"Session {session_id} not found for patient {patient_id}"}), 404
        except Exception as e:
            return jsonify({"error": f"Patient {patient_id} not found"}), 404

        # Save file to GridFS - Original logic preserved
        filename = secure_filename(file.filename)
        file_id = save_file_to_gridfs(file, filename)
        print(f"✓ Video saved to GridFS with ID: {file_id}")

        # Attach to session - Original logic preserved
        success = attach_video_to_session(patient_id, session_id, file_id)
        if not success:
            return jsonify({"error": "Failed to attach video to session"}), 500

        logger.info(f"Video uploaded successfully for speech analysis: {filename} -> {file_id}")
        return jsonify({
            "message": "Video uploaded successfully",
            "file_id": file_id,
            "patient_id": patient_id,
            "session_id": session_id
        }), 200

    except Exception as e:
        print(f"❌ Error uploading video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@speech_bp.route('/analyze/<file_id>/<int:patient_id>/<int:session_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def analyze_speech_and_tov(file_id, patient_id, session_id):
    """
    Analyze speech and tone of voice from uploaded video/audio
    Original logic preserved exactly with enhanced error handling
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"🔄 Starting speech analysis for file {file_id}, patient {patient_id}, session {session_id}")
        
        # Validate file_id format - Original logic preserved
        if not ObjectId.is_valid(file_id):
            return jsonify({"error": "Invalid file ID format"}), 400

        # Verify file exists in GridFS - Original logic preserved
        try:
            fs.get(ObjectId(file_id))
        except gridfs.errors.NoFile:
            return jsonify({"error": "File not found"}), 404

        # Verify patient and session exist - Original logic preserved
        try:
            patient = get_patient_by_id(patient_id)
            session_found = any(sess.session_id == session_id for sess in patient.sessions)
            if not session_found:
                return jsonify({"error": f"Session {session_id} not found for patient {patient_id}"}), 404
        except Exception as e:
            return jsonify({"error": f"Patient {patient_id} not found"}), 404

        # Process speech and TOV analysis - Original logic preserved
        try:
            print("🔄 Processing speech and tone of voice analysis...")
            results = process_speech_and_tov_by_id(file_id, patient_id, session_id)
            print(f"✓ Analysis completed successfully. Result: {results}")
            
            logger.info(f"Speech analysis completed for file {file_id}, patient {patient_id}, session {session_id}")
            return jsonify({
                "message": "Speech analysis completed successfully",
                "pdf_id": results,
                "patient_id": patient_id,
                "session_id": session_id,
                "file_id": file_id
            }), 200

        except Exception as e:
            print(f"❌ Error during speech analysis: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    except Exception as e:
        print(f"❌ Error in analyze_speech_and_tov: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@speech_bp.route('/download-report/<report_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_speech_report(report_id):
    """
    Download the generated speech analysis report
    Original logic preserved exactly with enhanced error handling
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"📥 Download request for speech report ID: {report_id}")
        
        # Validate ObjectId format - Original logic preserved
        if not ObjectId.is_valid(report_id):
            return jsonify({"error": "Invalid report ID format"}), 400

        # Get file from GridFS - Original logic preserved
        filename, file_data, content_type = download_report_by_id(report_id)
        
        if file_data is None:
            print(f"❌ Speech report not found: {report_id}")
            return jsonify({"error": f"Report not found: {report_id}"}), 404
        
        print(f"✓ Preparing speech report download: {filename} ({len(file_data)} bytes)")
        
        # Create a BytesIO object from the file data - Original logic preserved
        file_stream = BytesIO(file_data)
        
        # Check Flask version compatibility - Original logic preserved
        try:
            # Try new Flask syntax first
            logger.info(f"Speech report downloaded: {filename} ({report_id})")
            return send_file(
                file_stream,
                download_name=filename,
                as_attachment=True,
                mimetype=content_type or "application/pdf"
            )
        except TypeError:
            # Fall back to old Flask syntax
            logger.info(f"Speech report downloaded (legacy): {filename} ({report_id})")
            return send_file(
                file_stream,
                attachment_filename=filename,
                as_attachment=True,
                mimetype=content_type or "application/pdf"
            )
            
    except Exception as e:
        print(f"❌ Error downloading speech report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@speech_bp.route('/status/<int:patient_id>/<int:session_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_speech_analysis_status(patient_id, session_id):
    """
    Get the status of speech analysis for a specific patient session
    Original logic preserved exactly with enhanced error handling
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"📊 Checking speech analysis status for patient {patient_id}, session {session_id}")
        
        # Verify patient and session exist - Original logic preserved
        try:
            patient = get_patient_by_id(patient_id)
            target_session = None
            for sess in patient.sessions:
                if sess.session_id == session_id:
                    target_session = sess
                    break
            
            if not target_session:
                return jsonify({"error": f"Session {session_id} not found for patient {patient_id}"}), 404
                
        except Exception as e:
            return jsonify({"error": f"Patient {patient_id} not found"}), 404

        # Check speech analysis status - Original logic preserved
        speech_data = target_session.feature_data.get("Speech", {}) if target_session.feature_data else {}
        
        status = {
            "patient_id": patient_id,
            "session_id": session_id,
            "has_video": bool(target_session.video_files),
            "has_speech_excel": bool(speech_data.get("speech_excel")),
            "has_transcription": bool(target_session.text),
            "has_report": bool(target_session.report),
            "analysis_complete": bool(speech_data.get("speech_excel") and target_session.text)
        }
        
        print(f"✓ Speech analysis status: {status}")
        logger.info(f"Speech analysis status retrieved for patient {patient_id}, session {session_id}")
        return jsonify(status), 200

    except Exception as e:
        print(f"❌ Error checking speech analysis status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Status check failed: {str(e)}"}), 500

@speech_bp.route('/results/<int:patient_id>/<int:session_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_speech_analysis_results(patient_id, session_id):
    """
    Get the results of speech analysis for a specific patient session
    Original logic preserved exactly with enhanced error handling
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"📋 Getting speech analysis results for patient {patient_id}, session {session_id}")
        
        # Verify patient and session exist - Original logic preserved
        try:
            patient = get_patient_by_id(patient_id)
            target_session = None
            for sess in patient.sessions:
                if sess.session_id == session_id:
                    target_session = sess
                    break
            
            if not target_session:
                return jsonify({"error": f"Session {session_id} not found for patient {patient_id}"}), 404
                
        except Exception as e:
            return jsonify({"error": f"Patient {patient_id} not found"}), 404

        # Get speech analysis data - Original logic preserved
        speech_data = target_session.feature_data.get("Speech", {}) if target_session.feature_data else {}
        
        results = {
            "patient_id": patient_id,
            "session_id": session_id,
            "transcription": target_session.text,
            "speech_excel_id": str(speech_data.get("speech_excel")) if speech_data.get("speech_excel") else None,
            "report_id": str(target_session.report) if target_session.report else None,
            "video_file_id": str(target_session.video_files) if target_session.video_files else None,
            "analysis_date": target_session.date.isoformat() if target_session.date else None
        }
        
        print(f"✓ Speech analysis results retrieved: {results}")
        logger.info(f"Speech analysis results retrieved for patient {patient_id}, session {session_id}")
        return jsonify(results), 200

    except Exception as e:
        print(f"❌ Error getting speech analysis results: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Results retrieval failed: {str(e)}"}), 500

# ================================
# NEW ENHANCED ENDPOINTS (WHILE PRESERVING ORIGINAL FUNCTIONALITY)
# ================================

@speech_bp.route('/validate-file/<file_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def validate_speech_file(file_id):
    """Validate if file exists and is suitable for speech analysis"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        _validate_file_id_and_exists(file_id)
        
        return jsonify({
            "message": "File is valid for speech analysis",
            "file_id": file_id,
            "valid": True
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@speech_bp.route('/session-capabilities/<int:patient_id>/<int:session_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_session_speech_capabilities(patient_id, session_id):
    """Get speech analysis capabilities for a session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        patient = _validate_patient_and_session(patient_id, session_id)
        
        # Find the session
        target_session = None
        for sess in patient.sessions:
            if sess.session_id == session_id:
                target_session = sess
                break
        
        speech_data = target_session.feature_data.get("Speech", {}) if target_session.feature_data else {}
        
        capabilities = {
            "patient_id": patient_id,
            "session_id": session_id,
            "can_analyze": bool(target_session.video_files),
            "has_existing_analysis": bool(speech_data.get("speech_excel")),
            "has_transcription": bool(target_session.text),
            "video_file_id": str(target_session.video_files) if target_session.video_files else None,
            "speech_excel_id": str(speech_data.get("speech_excel")) if speech_data.get("speech_excel") else None,
            "supported_formats": list(ALLOWED_EXTENSIONS)
        }
        
        logger.info(f"Speech capabilities retrieved for patient {patient_id}, session {session_id}")
        return jsonify(capabilities), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# UTILITY ENDPOINTS
# ================================

@speech_bp.route('/health', methods=['GET', 'OPTIONS'])
@cross_origin()
def health_check():
    """Health check for speech analysis service"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Test GridFS connection
        fs.list()
        
        return jsonify({
            "status": "healthy",
            "service": "speech_controller",
            "gridfs_connected": True,
            "supported_formats": list(ALLOWED_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "features": {
                "speech_analysis": True,
                "tone_of_voice": True,
                "transcription": True,
                "report_generation": True
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "speech_controller",
            "error": str(e)
        }), 500

@speech_bp.route('/supported-formats', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_supported_formats():
    """Get supported file formats for speech analysis"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        video_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        audio_formats = ['mp3', 'wav', 'flac']
        
        return jsonify({
            "supported_formats": {
                "all": list(ALLOWED_EXTENSIONS),
                "video": video_formats,
                "audio": audio_formats
            },
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "recommendations": {
                "best_video_format": "mp4",
                "best_audio_format": "wav",
                "optimal_duration": "2-30 minutes"
            }
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# ERROR HANDLERS (ORIGINAL LOGIC PRESERVED)
# ================================

@speech_bp.errorhandler(413)
def file_too_large(error):
    """Original error handler preserved"""
    return jsonify({"error": "File too large. Maximum size allowed is determined by server configuration."}), 413

@speech_bp.errorhandler(400)
def bad_request(error):
    """Original error handler preserved"""
    return jsonify({"error": "Bad request. Please check your input parameters."}), 400

@speech_bp.errorhandler(404)
def not_found(error):
    """Original error handler preserved"""
    return jsonify({"error": "Resource not found."}), 404

@speech_bp.errorhandler(500)
def internal_error(error):
    """Original error handler preserved"""
    return jsonify({"error": "Internal server error. Please try again later."}), 500