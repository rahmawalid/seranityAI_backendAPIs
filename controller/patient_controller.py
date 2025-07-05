"""
Patient Controller - HTTP Request Handler Layer
Handles all HTTP requests for patient-related functionality
Updated to align with the new ReportService architecture
"""

import datetime
import os
import tempfile
import logging
from flask import Blueprint, request, jsonify, make_response, Response, send_file
from flask_cors import cross_origin
from bson import ObjectId
from mongoengine import DoesNotExist, ValidationError
from werkzeug.utils import secure_filename
from io import BytesIO

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Service Layer Imports (Controllers should use services, not repositories directly)
from services.FER_service import run_fer_on_video_by_id  # Legacy function for now
from services.report_service import ReportService
from services.doctor_notes_service import DoctorNotesService

# Repository Layer Imports (only when services don't exist yet)
from repository.patient_repository import (
    create_patient,
    get_patient_by_id,
    update_patient,
    delete_patient,
    list_patients_by_doctor,
    add_session_to_patient,
    save_audio_to_gridfs,
    attach_audio_to_session,
    save_video_to_gridfs,
    attach_video_to_session,
    save_pdf_to_gridfs,
    attach_pdf_to_session,
    save_excel_to_gridfs,
    attach_model_file,
    save_fer_video_to_session,
    download_report_by_id,
)

from controller.doctor_notes_controller import (
    generate_enhanced_report
)

# File Repository for GridFS operations
from repository.file_repository import (
    get_file_from_gridfs,
    get_file_content_from_gridfs,
    save_file_to_gridfs,
)

# Model imports
from model.patient_model import Patient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
patient_bp = Blueprint("patient", __name__)

# Initialize services
try:
    report_service = ReportService()
    logger.info("✓ ReportService initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ReportService: {e}")
    report_service = None

try:
    doctor_notes_service = DoctorNotesService()
    logger.info("✓ DoctorNotesService initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize DoctorNotesService: {e}")
    doctor_notes_service = None

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
    logger.error(f"Controller error: {error}")
    
    if isinstance(error, DoesNotExist):
        return jsonify({"error": "Resource not found", "details": str(error)}), 404
    elif isinstance(error, ValidationError):
        return jsonify({"error": "Validation error", "details": str(error)}), 400
    elif isinstance(error, ValueError):
        return jsonify({"error": "Invalid input", "details": str(error)}), 400
    else:
        return jsonify({"error": "Internal server error", "details": str(error)}), default_status

def _validate_file_upload(file, allowed_extensions, max_size_mb=50):
    """Validate file upload with proper error messages"""
    if not file or file.filename == "":
        raise ValueError("No file provided")
    
    filename = secure_filename(file.filename)
    if not filename:
        raise ValueError("Invalid filename")
    
    # Check extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise ValueError(f"File type {ext} not allowed. Allowed types: {allowed_extensions}")
    
    # Check file size if possible (some file objects don't support this)
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

def convert_bson(obj):
    """Convert BSON objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: convert_bson(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bson(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    else:
        return obj

def mongo_to_dict_patient(doc):
    """Convert Patient document to dictionary"""
    try:
        data = doc.to_mongo().to_dict()
        data["_id"] = str(data["_id"])
        data = convert_bson(data)
        return data
    except Exception as e:
        logger.error(f"Error converting patient to dict: {e}")
        raise ValueError(f"Failed to convert patient data: {str(e)}")

def mongo_to_dict_session(sess):
    """Convert Session embedded document to dictionary"""
    try:
        data = sess.to_mongo().to_dict()
        
        # Handle date conversion
        if "date" in data and isinstance(data["date"], datetime.datetime):
            data["date"] = data["date"].isoformat()
        
        # Convert ObjectIds in common fields
        for key in ("audioFiles", "videoFiles", "report", "transcription"):
            if key in data and isinstance(data[key], ObjectId):
                data[key] = str(data[key])
        
        # Convert ObjectIds in model_files
        if "model_files" in data and isinstance(data["model_files"], dict):
            for k, v in data["model_files"].items():
                if isinstance(v, dict):
                    for nested_k, nested_v in v.items():
                        if isinstance(nested_v, ObjectId):
                            v[nested_k] = str(nested_v)
                elif isinstance(v, ObjectId):
                    data["model_files"][k] = str(v)
        
        # Convert ObjectIds in feature_data (for new report service)
        if "feature_data" in data and isinstance(data["feature_data"], dict):
            for feature_type, feature_info in data["feature_data"].items():
                if isinstance(feature_info, dict):
                    for k, v in feature_info.items():
                        if isinstance(v, ObjectId):
                            feature_info[k] = str(v)
                        elif isinstance(v, list):
                            feature_info[k] = [str(item) if isinstance(item, ObjectId) else item for item in v]
        
        # Convert ObjectIds in doctor_notes_images
        if "doctor_notes_images" in data and isinstance(data["doctor_notes_images"], list):
            data["doctor_notes_images"] = [str(oid) for oid in data["doctor_notes_images"]]
        
        return data
    except Exception as e:
        logger.error(f"Error converting session to dict: {e}")
        raise ValueError(f"Failed to convert session data: {str(e)}")

# ================================
# PATIENT CRUD ENDPOINTS
# ================================

@patient_bp.route("/create-patient", methods=["POST", "OPTIONS"])
@cross_origin()
def create_patient_controller():
    """Create a new patient"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        patient_data = request.get_json()
        if not patient_data:
            raise ValueError("No patient data provided")
        
        created_patient = create_patient(patient_data)
        
        logger.info(f"Patient created successfully: {created_patient.patientID}")
        return jsonify({
            "message": "Patient created successfully",
            "patient_id": created_patient.patientID,
            "patient": mongo_to_dict_patient(created_patient)
        }), 201
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/get-patient/<int:patient_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def get_patient_controller(patient_id):
    """Get patient by ID"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        patient = get_patient_by_id(patient_id)
        return jsonify(mongo_to_dict_patient(patient)), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/update-patient/<int:patient_id>", methods=["PUT", "OPTIONS"])
@cross_origin()
def update_patient_controller(patient_id):
    """Update patient information"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        updated_data = request.get_json()
        if not updated_data:
            raise ValueError("No update data provided")
        
        patient = update_patient(patient_id, updated_data)
        
        logger.info(f"Patient updated successfully: {patient_id}")
        return jsonify({
            "message": "Patient updated successfully",
            "patient": mongo_to_dict_patient(patient)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/delete-patient/<int:patient_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_patient_controller(patient_id):
    """Delete patient"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        delete_patient(patient_id)
        
        logger.info(f"Patient deleted successfully: {patient_id}")
        return jsonify({"message": "Patient deleted successfully"}), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/list-patients", methods=["GET", "OPTIONS"])
@cross_origin()
def list_patients_controller():
    """List all patients"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        patients = Patient.objects()
        patients_list = [mongo_to_dict_patient(p) for p in patients]
        
        logger.info(f"Listed {len(patients_list)} patients")
        return jsonify({
            "patients": patients_list,
            "count": len(patients_list)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/<string:doctor_id>/patients", methods=["GET", "OPTIONS"])
@cross_origin()
def list_patients_by_doctor_controller(doctor_id):
    """List patients by doctor ID"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        patients = list_patients_by_doctor(doctor_id)
        patients_list = [mongo_to_dict_patient(p) for p in patients]
        
        logger.info(f"Listed {len(patients_list)} patients for doctor {doctor_id}")
        return jsonify({
            "patients": patients_list,
            "doctor_id": doctor_id,
            "count": len(patients_list)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# SESSION MANAGEMENT ENDPOINTS
# ================================

@patient_bp.route("/<int:patient_id>/sessions", methods=["POST", "OPTIONS"])
@cross_origin()
def add_session_controller(patient_id):
    """Add a new session to patient"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        session_data = request.get_json()
        if not session_data:
            raise ValueError("No session data provided")
        
        new_session = add_session_to_patient(patient_id, session_data)
        
        logger.info(f"Session added successfully to patient {patient_id}: session {new_session.session_id}")
        return jsonify({
            "message": "Session added successfully",
            "session": mongo_to_dict_session(new_session),
            "patient_id": patient_id,
            "session_id": new_session.session_id
        }), 201
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# FILE UPLOAD ENDPOINTS
# ================================

@patient_bp.route("/upload-audio/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def upload_audio_file(patient_id, session_id):
    """Upload audio file to session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        file = request.files.get("file")
        filename = _validate_file_upload(file, [".mp3", ".wav", ".m4a"], max_size_mb=100)
        
        logger.info(f"Uploading audio file: {filename}")
        file_id = save_audio_to_gridfs(file)
        
        if attach_audio_to_session(patient_id, session_id, file_id):
            logger.info(f"Audio uploaded successfully: {filename} -> {file_id}")
            return jsonify({
                "message": "Audio uploaded successfully",
                "file_id": file_id,
                "filename": filename,
                "patient_id": patient_id,
                "session_id": session_id
            }), 200
        else:
            raise ValueError("Failed to attach audio to session")
            
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/upload-video/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def upload_video_file(patient_id, session_id):
    """Upload video file to session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        file = request.files.get("file")
        filename = _validate_file_upload(file, [".mp4", ".mov", ".avi"], max_size_mb=500)
        
        logger.info(f"Uploading video file: {filename}")
        file_id = save_video_to_gridfs(file)
        
        if attach_video_to_session(patient_id, session_id, file_id):
            logger.info(f"Video uploaded successfully: {filename} -> {file_id}")
            return jsonify({
                "message": "Video uploaded successfully",
                "file_id": file_id,
                "filename": filename,
                "patient_id": patient_id,
                "session_id": session_id
            }), 200
        else:
            raise ValueError("Failed to attach video to session")
            
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/upload-report/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def upload_report_file(patient_id, session_id):
    """Upload PDF report to session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        file = request.files.get("file")
        filename = _validate_file_upload(file, [".pdf"], max_size_mb=50)
        
        logger.info(f"Uploading report file: {filename}")
        file_id = save_pdf_to_gridfs(file)
        
        if attach_pdf_to_session(patient_id, session_id, file_id):
            logger.info(f"Report uploaded successfully: {filename} -> {file_id}")
            return jsonify({
                "message": "Report uploaded successfully",
                "file_id": file_id,
                "filename": filename,
                "patient_id": patient_id,
                "session_id": session_id
            }), 200
        else:
            raise ValueError("Failed to attach report to session")
            
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/upload-model-file/<int:patient_id>/<int:session_id>/<model_type>/<file_label>", methods=["POST", "OPTIONS"])
@cross_origin()
def upload_model_file(patient_id, session_id, model_type, file_label):
    """Upload model file (Excel, CSV, etc.) to session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        file = request.files.get("file")
        filename = _validate_file_upload(file, [".xlsx", ".xls", ".csv"], max_size_mb=25)
        
        logger.info(f"Uploading model file: {filename} -> {model_type}/{file_label}")
        file_id = save_excel_to_gridfs(file)
        
        if attach_model_file(patient_id, session_id, model_type, file_label, file_id):
            logger.info(f"Model file uploaded successfully: {filename} -> {file_id}")
            return jsonify({
                "message": "Model file uploaded successfully",
                "file_id": file_id,
                "filename": filename,
                "model_type": model_type,
                "file_label": file_label,
                "patient_id": patient_id,
                "session_id": session_id
            }), 200
        else:
            raise ValueError("Failed to attach model file to session")
            
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# FER ANALYSIS ENDPOINTS
# ================================

@patient_bp.route("/<int:patient_id>/sessions/<int:session_id>/upload-fer-video", methods=["POST", "OPTIONS"])
@cross_origin()
def upload_fer_video(patient_id, session_id):
    """Upload video specifically for FER analysis"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if "video" not in request.files:
            raise ValueError("No video file uploaded")
        
        video_file = request.files["video"]
        filename = _validate_file_upload(video_file, [".mp4", ".mov", ".avi"], max_size_mb=500)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, filename)
            video_file.save(video_path)
            
            logger.info(f"Uploading FER video: {filename}")
            video_file_id = save_fer_video_to_session(patient_id, session_id, video_path, filename)
            
            logger.info(f"FER video uploaded successfully: {filename} -> {video_file_id}")
            return jsonify({
                "message": "Video uploaded and saved for FER analysis",
                "video_file_id": str(video_file_id),
                "filename": filename,
                "patient_id": patient_id,
                "session_id": session_id
            }), 200
            
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/fer/analyze/<string:file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def analyze_fer_basic(file_id):
    """Basic FER analysis (video ID only)"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        logger.info(f"Starting basic FER analysis for file: {file_id}")
        
        # TODO: Update to use FER service class when available
        df = run_fer_on_video_by_id(file_id)
        
        return jsonify({
            "message": "FER analysis completed",
            "file_id": file_id,
            "emotions": df.to_dict(orient="records") if df is not None else [],
            "total_frames": len(df) if df is not None else 0
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/fer/analyze/<string:file_id>/<int:patient_id>/<int:session_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def analyze_fer_and_save(file_id, patient_id, session_id):
    """Full FER analysis with results saved to session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        logger.info(f"Starting FER analysis for file: {file_id}, patient: {patient_id}, session: {session_id}")
        
        # TODO: Update to use FER service class when available
        df, excel_file_id, plot_ids = run_fer_on_video_by_id(file_id, patient_id, session_id)
        
        logger.info(f"FER analysis completed: {len(plot_ids)} plots generated")
        return jsonify({
            "message": "FER analysis completed and saved",
            "file_id": file_id,
            "patient_id": patient_id,
            "session_id": session_id,
            "excel_file_id": str(excel_file_id),
            "plot_image_ids": [str(pid) for pid in plot_ids],
            "emotions": df.to_dict(orient="records") if df is not None else [],
            "total_frames": len(df) if df is not None else 0,
            "plots_count": len(plot_ids)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# FILE STREAMING ENDPOINTS
# ================================

@patient_bp.route("/stream-audio/<file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def stream_audio(file_id):
    """Stream audio file"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        filename, content, content_type = get_file_content_from_gridfs(file_id)
        
        if not content_type.startswith('audio/'):
            content_type = "audio/mpeg"  # Default fallback
        
        return Response(content, mimetype=content_type)
        
    except Exception as e:
        logger.error(f"Error streaming audio {file_id}: {e}")
        return jsonify({"error": "Audio not found", "file_id": file_id}), 404

@patient_bp.route("/stream-video/<file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def stream_video(file_id):
    """Stream video file"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        filename, content, content_type = get_file_content_from_gridfs(file_id)
        
        if not content_type.startswith('video/'):
            content_type = "video/mp4"  # Default fallback
        
        return Response(content, mimetype=content_type)
        
    except Exception as e:
        logger.error(f"Error streaming video {file_id}: {e}")
        return jsonify({"error": "Video not found", "file_id": file_id}), 404

@patient_bp.route("/view-report/<file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def view_report(file_id):
    """View PDF report"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        filename, content, content_type = get_file_content_from_gridfs(file_id)
        
        if not content_type.startswith('application/pdf'):
            content_type = "application/pdf"  # Default fallback
        
        return Response(content, mimetype=content_type)
        
    except Exception as e:
        logger.error(f"Error viewing report {file_id}: {e}")
        return jsonify({"error": "Report not found", "file_id": file_id}), 404

# ================================
# ADVANCED REPORT ENDPOINTS (Using ReportService)
# ================================

@patient_bp.route("/reports/generate/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_analysis_report(patient_id, session_id):
    """
    Generate analysis report for session using the new ReportService
    Automatically determines whether to use TOV-only or comprehensive analysis
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        logger.info(f"Generating analysis report for patient {patient_id}, session {session_id}")
        
        # Use the report service to generate the report
        report_data = report_service.generate_session_analysis_report(patient_id, session_id)
        
        logger.info(f"Analysis report generated successfully: {report_data['report_id']}")
        return jsonify({
            "message": "Analysis report generated successfully",
            "success": True,
            "report_id": report_data["report_id"],
            "patient_id": patient_id,
            "session_id": session_id,
            "analysis_type": "automatic",  # Service determines TOV-only vs comprehensive
            "fer_data_available": bool(report_data.get("fer_excel")),
            "speech_data_available": bool(report_data.get("tov_excel")),
            "fer_images_count": len(report_data.get("fer_images", [])),
            "text_available": bool(report_data.get("text"))
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/reports/metadata/<int:patient_id>/<int:session_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def get_report_metadata(patient_id, session_id):
    """Get report metadata and analysis capabilities for session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        metadata = report_service.get_report_metadata(patient_id, session_id)
        
        return jsonify({
            "message": "Report metadata retrieved successfully",
            "success": True,
            "metadata": metadata,
            "analysis_recommendation": {
                "can_generate": metadata["can_generate_report"],
                "has_fer": metadata["available_data"]["fer_excel"],
                "has_speech": metadata["available_data"]["speech_excel"],
                "has_images": metadata["available_data"]["fer_images"],
                "recommended_type": (
                    "comprehensive" if metadata["available_data"]["fer_excel"] and metadata["available_data"]["speech_excel"]
                    else "tov_only" if metadata["available_data"]["speech_excel"]
                    else "insufficient_data"
                )
            }
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/reports/download/<int:patient_id>/<int:session_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def download_session_report(patient_id, session_id):
    """Download existing report or generate new one if none exists"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        logger.info(f"Downloading report for patient {patient_id}, session {session_id}")
        
        # Try to download existing report or generate new one
        filename, content, content_type = report_service.download_session_report(patient_id, session_id)
        
        if not content:
            raise ValueError("Could not generate or retrieve report")
        
        logger.info(f"Report downloaded successfully: {filename}")
        return send_file(
            BytesIO(content),
            mimetype=content_type or "application/pdf",
            as_attachment=True,
            download_name=filename or f"analysis_report_p{patient_id}_s{session_id}.pdf"
        )
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/reports/force-regenerate/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def force_regenerate_report(patient_id, session_id):
    """Force regenerate report even if one already exists"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        logger.info(f"Force regenerating report for patient {patient_id}, session {session_id}")
        
        # Force regeneration by directly calling the service
        report_data = report_service.generate_session_analysis_report(patient_id, session_id)
        
        logger.info(f"Report force-regenerated successfully: {report_data['report_id']}")
        return jsonify({
            "message": "Report regenerated successfully",
            "success": True,
            "report_id": report_data["report_id"],
            "patient_id": patient_id,
            "session_id": session_id,
            "regenerated": True,
            "analysis_type": "forced_regeneration"
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# LEGACY REPORT ENDPOINTS (Backward Compatibility)
# ================================

@patient_bp.route("/report/download/<report_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def download_pdf_report_legacy(report_id):
    """Download PDF report by ID using legacy repository method"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        logger.info(f"Downloading legacy report: {report_id}")
        
        filename, content, content_type = download_report_by_id(report_id)
        
        if not content:
            raise ValueError("Report not found or empty")
        
        return send_file(
            BytesIO(content),
            mimetype=content_type or "application/pdf",
            as_attachment=True,
            attachment_filename=filename or f"report_{report_id}.pdf"
        )
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/report/generate/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_session_report_legacy(patient_id, session_id):
    """Legacy endpoint - redirects to new report generation"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        logger.warning("Using legacy report generation endpoint. Consider using /reports/generate/ instead.")
        
        if not report_service:
            raise ValueError("ReportService not available")
        
        # Redirect to new service
        report_data = report_service.generate_session_analysis_report(patient_id, session_id)
        
        return jsonify({
            "message": "Report generated successfully (legacy endpoint)",
            "report_id": report_data["report_id"],
            "patient_id": patient_id,
            "session_id": session_id,
            "legacy_endpoint": True,
            "recommendation": "Use /reports/generate/ for new features"
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# SPECIALIZED REPORT ENDPOINTS
# ================================

@patient_bp.route("/reports/tov-only/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_tov_only_report(patient_id, session_id):
    """Generate TOV-only analysis report specifically"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        logger.info(f"Generating TOV-only report for patient {patient_id}, session {session_id}")
        
        # First check if we have the required data
        metadata = report_service.get_report_metadata(patient_id, session_id)
        
        if not metadata["available_data"]["speech_excel"]:
            raise ValueError("No speech/TOV analysis data available for this session")
        
        # Generate report (service will automatically detect TOV-only mode)
        report_data = report_service.generate_session_analysis_report(patient_id, session_id)
        
        logger.info(f"TOV-only report generated: {report_data['report_id']}")
        return jsonify({
            "message": "TOV-only analysis report generated successfully",
            "success": True,
            "report_id": report_data["report_id"],
            "patient_id": patient_id,
            "session_id": session_id,
            "analysis_type": "tov_only",
            "prompt_used": "text_tone_analysis_only",
            "has_fer_data": bool(report_data.get("fer_excel")),
            "speech_chunks_analyzed": "Available in report"
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/reports/comprehensive/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_comprehensive_report(patient_id, session_id):
    """Generate comprehensive FER+TOV analysis report specifically"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        logger.info(f"Generating comprehensive report for patient {patient_id}, session {session_id}")
        
        # Check if we have the required data for comprehensive analysis
        metadata = report_service.get_report_metadata(patient_id, session_id)
        
        if not metadata["available_data"]["speech_excel"]:
            raise ValueError("No speech/TOV analysis data available for comprehensive analysis")
        
        if not metadata["available_data"]["fer_excel"]:
            raise ValueError("No FER analysis data available for comprehensive analysis")
        
        # Generate comprehensive report
        report_data = report_service.generate_session_analysis_report(patient_id, session_id)
        
        logger.info(f"Comprehensive report generated: {report_data['report_id']}")
        return jsonify({
            "message": "Comprehensive FER+TOV analysis report generated successfully",
            "success": True,
            "report_id": report_data["report_id"],
            "patient_id": patient_id,
            "session_id": session_id,
            "analysis_type": "comprehensive",
            "prompt_used": "combined_fer_and_tov_analysis",
            "fer_images_count": len(report_data.get("fer_images", [])),
            "has_mismatch_analysis": True,
            "individual_fer_graphs": True
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/reports/capabilities/<int:patient_id>/<int:session_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def get_analysis_capabilities(patient_id, session_id):
    """Get detailed analysis capabilities and recommendations"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        metadata = report_service.get_report_metadata(patient_id, session_id)
        
        # Determine available analysis types
        capabilities = {
            "tov_only": {
                "available": metadata["available_data"]["speech_excel"],
                "description": "Text tone analysis with DSM-5/ICD-11 diagnostic insights",
                "requirements": ["Speech analysis data"],
                "prompt_type": "tov_only_diagnostic_prompt"
            },
            "comprehensive": {
                "available": metadata["available_data"]["fer_excel"] and metadata["available_data"]["speech_excel"],
                "description": "Combined facial expression and text tone analysis with mismatch detection",
                "requirements": ["FER analysis data", "Speech analysis data"],
                "prompt_type": "comprehensive_diagnostic_prompt"
            },
            "fer_only": {
                "available": metadata["available_data"]["fer_excel"] and not metadata["available_data"]["speech_excel"],
                "description": "Facial expression analysis only (limited insights)",
                "requirements": ["FER analysis data"],
                "prompt_type": "not_implemented"
            }
        }
        
        # Determine best recommendation
        if capabilities["comprehensive"]["available"]:
            recommended = "comprehensive"
            reason = "Both FER and speech data available - provides most detailed analysis"
        elif capabilities["tov_only"]["available"]:
            recommended = "tov_only"
            reason = "Speech data available - provides good clinical insights"
        else:
            recommended = "none"
            reason = "Insufficient data for analysis"
        
        return jsonify({
            "message": "Analysis capabilities retrieved",
            "success": True,
            "patient_id": patient_id,
            "session_id": session_id,
            "capabilities": capabilities,
            "recommendation": {
                "type": recommended,
                "reason": reason
            },
            "metadata": metadata,
            "available_prompts": {
                "tov_only": "Advanced AI clinical psychologist analyzing through text tone analysis",
                "comprehensive": "Advanced AI clinical psychologist analyzing through combined text tone and facial expression analysis"
            }
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# REPORT STATUS AND MANAGEMENT
# ================================

@patient_bp.route("/reports/status/<int:patient_id>/<int:session_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def get_report_status(patient_id, session_id):
    """Get current report status for session"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        metadata = report_service.get_report_metadata(patient_id, session_id)
        
        # Determine status
        if metadata["has_existing_report"]:
            status = "completed"
            message = "Report already exists and is ready for download"
        elif metadata["can_generate_report"]:
            status = "ready_to_generate"
            message = "Sufficient data available to generate report"
        else:
            status = "insufficient_data"
            message = "Insufficient analysis data to generate report"
        
        return jsonify({
            "message": message,
            "success": True,
            "patient_id": patient_id,
            "session_id": session_id,
            "status": status,
            "existing_report_id": metadata.get("existing_report_id"),
            "can_generate": metadata["can_generate_report"],
            "data_summary": {
                "fer_excel": metadata["available_data"]["fer_excel"],
                "fer_images": metadata["available_data"]["fer_images"],
                "speech_excel": metadata["available_data"]["speech_excel"],
                "session_text": metadata["available_data"]["session_text"]
            }
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# DOCTOR NOTES INTEGRATION
# ================================

@patient_bp.route("/reports/with-doctor-notes/<int:patient_id>/<int:session_id>", methods=["POST", "OPTIONS"])
@cross_origin()
def generate_report_with_doctor_notes(patient_id, session_id):
    """Generate enhanced report with doctor notes integration"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not doctor_notes_service:
            raise ValueError("DoctorNotesService not available")
        
        logger.info(f"Generating enhanced report with doctor notes for patient {patient_id}, session {session_id}")
        
        # Use doctor notes service for enhanced analysis
        result, status_code = generate_enhanced_report(patient_id, session_id)
        print("report generation")
        print(result)
        print(result)

        if not result["success"]:
            raise ValueError(result.get("error", "Failed to generate enhanced report"))
        
        return jsonify({
            "message": "Enhanced report with doctor notes generated successfully",
            "success": True,
            "report_id": result["data"]["report_id"],
            "patient_id": patient_id,
            "session_id": session_id,
            "analysis_type": result["data"]["analysis_type"],
            "doctor_notes_count": result["data"]["doctor_notes_count"],
            "prompt_used": result["data"]["prompt_used"],
            "images_included": result["data"]["images_included"]
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# DEBUGGING AND TESTING ENDPOINTS
# ================================

@patient_bp.route("/reports/debug/<int:patient_id>/<int:session_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def debug_report_data(patient_id, session_id):
    """Debug endpoint to check available data for report generation"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not report_service:
            raise ValueError("ReportService not available")
        
        from repository.patient_repository import _get_patient_by_id, _find_session_index
        
        # Get raw data
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        session = patient.sessions[session_index]
        
        # Extract feature files data
        feature_files_data = report_service._extract_session_feature_files(session)
        
        debug_info = {
            "patient_id": patient_id,
            "session_id": session_id,
            "session_data": mongo_to_dict_session(session),
            "feature_files": {
                "fer_excel_id": feature_files_data.get("fer_excel_id"),
                "fer_excel_available": feature_files_data.get("fer_excel_file") is not None,
                "fer_images_count": len(feature_files_data.get("fer_images", [])),
                "tov_excel_id": feature_files_data.get("tov_excel_id"),
                "tov_excel_available": feature_files_data.get("tov_excel_file") is not None
            },
            "validation": {
                "can_generate_tov_only": feature_files_data.get("tov_excel_id") is not None,
                "can_generate_comprehensive": (
                    feature_files_data.get("fer_excel_id") is not None and 
                    feature_files_data.get("tov_excel_id") is not None
                )
            }
        }
        
        return jsonify({
            "message": "Debug information retrieved",
            "success": True,
            "debug_info": debug_info,
            "warnings": []
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@patient_bp.route("/reports/test-prompts", methods=["GET", "OPTIONS"])
@cross_origin()
def test_report_prompts():
    """Test endpoint to verify prompt alignment"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        prompt_info = {
            "tov_only_prompt": {
                "description": "Advanced AI clinical psychologist analyzing patient-therapist sessions through text tone analysis",
                "sections": [
                    "BACKGROUND CLINICAL INFORMATION",
                    "SESSION INFORMATION", 
                    "TEXT TONE ANALYSIS"
                ],
                "analysis_instructions": [
                    "Emotional Pattern Analysis",
                    "Clinical Significance Evaluation", 
                    "Diagnostic Insights",
                    "Treatment Recommendations",
                    "Risk Assessment"
                ],
                "requirements": ["DSM-5/ICD-11 terminology", "Evidence-based analysis", "Direct clinical start"],
                "source": "tov_wirhout_notes.ipynb"
            },
            "comprehensive_prompt": {
                "description": "Advanced AI clinical psychologist analyzing through combined text tone and facial expression analysis",
                "sections": [
                    "BACKGROUND CLINICAL INFORMATION",
                    "SESSION INFORMATION",
                    "VIDEO/AUDIO TRANSCRIPT ANALYSIS", 
                    "INDIVIDUAL FER GRAPH ANALYSIS",
                    "FER INSIGHTS"
                ],
                "analysis_instructions": [
                    "Emotional Pattern Analysis (with verbal/non-verbal alignment)",
                    "Clinical Significance Evaluation",
                    "Diagnostic Insights", 
                    "Treatment Recommendations",
                    "Risk Assessment"
                ],
                "requirements": ["DSM-5/ICD-11 terminology", "Evidence-based analysis", "Mismatch detection", "Direct clinical start"],
                "source": "full_without_doctornotes_Apis.ipynb"
            }
        }
        
        return jsonify({
            "message": "Report prompts information",
            "success": True,
            "prompts": prompt_info,
            "service_status": {
                "report_service_available": report_service is not None,
                "doctor_notes_service_available": doctor_notes_service is not None
            }
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# HEALTH CHECK ENDPOINT
# ================================

@patient_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with service status"""
    try:
        # Basic connectivity test
        patient_count = Patient.objects.count()
        
        service_status = {
            "report_service": {
                "available": report_service is not None,
                "prompts": ["tov_only", "comprehensive"] if report_service else []
            },
            "doctor_notes_service": {
                "available": doctor_notes_service is not None,
                "enhanced_reports": doctor_notes_service is not None
            }
        }
        
        return jsonify({
            "status": "healthy",
            "service": "patient_controller",
            "patient_count": patient_count,
            "services": service_status,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "features": {
                "report_generation": report_service is not None,
                "tov_only_analysis": report_service is not None,
                "comprehensive_analysis": report_service is not None,
                "doctor_notes_integration": doctor_notes_service is not None
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "patient_controller",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 500