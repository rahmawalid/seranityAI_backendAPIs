"""
Doctor Controller - HTTP Request Handler Layer
Handles all HTTP requests for doctor-related functionality
Cleaned up to remove verification document functionality
"""

import datetime
import os
import logging
import gridfs
from datetime import timedelta
from flask import Blueprint, request, jsonify, Response, send_file
from flask_cors import cross_origin
from bson import ObjectId
from mongoengine import DoesNotExist, ValidationError
from werkzeug.utils import secure_filename
from io import BytesIO
from pymongo import MongoClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

# Repository Layer Imports (Controllers should use repositories for doctor management)
from repository.doctor_repository import (
    create_doctor,
    get_doctor_by_id,
    update_doctor,
    delete_doctor,
    login_doctor,
    update_doctor_password,
    verify_doctor_email,
    resend_verification_email,
    schedule_session_for_doctor,
    get_scheduled_sessions_for_doctor,
    save_file_to_gridfs,
    check_token_validity,
)
from repository.patient_repository import (
    get_patient_by_id,
    list_patients_by_doctor,
    create_patient_for_doctor,
)

# File Repository for GridFS operations
from repository.file_repository import (
    get_file_content_from_gridfs,
    save_file_to_gridfs as file_repo_save,
    check_file_exists_in_gridfs,
)

# Model imports
from model.doctor_model import Doctor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
doctor_blueprint = Blueprint("doctor_blueprint", __name__)

# File type configurations
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# ================================
# HELPER FUNCTIONS
# ================================

def _cors_preflight():
    """Handle CORS preflight requests"""
    from flask import make_response
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

def _handle_controller_error(error, default_status=500):
    """Consistent error handling for controller endpoints"""
    logger.error(f"Doctor controller error: {error}")
    
    if isinstance(error, DoesNotExist):
        return jsonify({"error": "Resource not found", "details": str(error)}), 404
    elif isinstance(error, ValidationError):
        return jsonify({"error": "Validation error", "details": str(error)}), 400
    elif isinstance(error, ValueError):
        return jsonify({"error": "Invalid input", "details": str(error)}), 400
    else:
        return jsonify({"error": "Internal server error", "details": str(error)}), default_status

def _validate_file_upload(file, allowed_extensions, max_size_mb=10):
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

def mongo_to_dict(doc):
    """Convert doctor document to dictionary with proper field handling"""
    try:
        data = doc.to_mongo().to_dict()
        data["_id"] = str(data["_id"])
        
        # Handle profile picture
        if "personal_info" in data and data["personal_info"].get("profile_picture"):
            data["personal_info"]["profile_picture"] = str(
                data["personal_info"]["profile_picture"]
            )
        
        # Handle scheduled sessions
        sessions = data.get("scheduledSessions", [])
        if sessions:
            for s in sessions:
                # Format the datetime field to ISO string
                d = s.get("datetime")
                if isinstance(d, datetime.datetime):
                    s["datetime"] = d.isoformat()
            data["scheduledSessions"] = sessions
        
        # Remove password field for security
        data.pop("password", None)
        
        return data
    except Exception as e:
        logger.error(f"Error converting doctor to dict: {e}")
        raise ValueError(f"Failed to convert doctor data: {str(e)}")

# ================================
# DOCTOR AUTHENTICATION ENDPOINTS
# ================================

@doctor_blueprint.route("/create-doctor", methods=["POST", "OPTIONS"])
@cross_origin()
def create_doctor_controller():
    """Create a new doctor account"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        doctor_data = request.get_json()
        if not doctor_data:
            raise ValueError("No doctor data provided")
        
        doctor = create_doctor(doctor_data)
        
        logger.info(f"Doctor created successfully: {doctor.doctor_ID}")
        return jsonify({
            "message": "Doctor created successfully. Please check your email for verification.",
            "doctor_ID": doctor.doctor_ID,
            "email_verified": doctor.email_verified
        }), 201
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/login-doctor", methods=["POST", "OPTIONS"])
@cross_origin()
def login_doctor_controller():
    """Authenticate doctor login"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or "email" not in data or "password" not in data:
            raise ValueError("Email and password are required")
        
        doctor = login_doctor(data["email"], data["password"])
        
        logger.info(f"Doctor logged in successfully: {doctor.doctor_ID}")
        return jsonify({
            "message": "Login successful",
            "doctor_ID": doctor.doctor_ID,
            "doctor_info": mongo_to_dict(doctor),
            "email_verified": doctor.email_verified
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e, 401)

@doctor_blueprint.route("/verify-email/<token>", methods=["GET", "OPTIONS"])
@cross_origin()
def verify_email_controller(token):
    """Verify doctor email using token"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not token:
            raise ValueError("Verification token is required")
        
        # Check token validity first
        token_status = check_token_validity(token)
        if not token_status["valid"]:
            raise ValueError(token_status["reason"])
        
        doctor = verify_doctor_email(token)
        
        logger.info(f"Email verified successfully for doctor: {doctor.doctor_ID}")
        return jsonify({
            "message": "✅ Email verified successfully",
            "doctor_ID": doctor.doctor_ID,
            "email": doctor.personal_info.email if doctor.personal_info else None
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/resend-verification", methods=["POST", "OPTIONS"])
@cross_origin()
def resend_verification_email_controller():
    """Resend verification email"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        data = request.get_json()
        email = data.get("email")
        
        if not email:
            raise ValueError("Email is required")
        
        success = resend_verification_email(email)
        
        if success:
            logger.info(f"Verification email resent to: {email}")
            return jsonify({
                "message": "Verification email sent successfully"
            }), 200
        else:
            raise ValueError("Failed to send verification email")
            
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# DOCTOR MANAGEMENT ENDPOINTS
# ================================

@doctor_blueprint.route("/get-doctor/<doctor_ID>", methods=["GET", "OPTIONS"])
@cross_origin()
def get_doctor_controller(doctor_ID):
    """Get doctor information by ID"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        doctor = get_doctor_by_id(doctor_ID)
        
        return jsonify({
            "doctor": mongo_to_dict(doctor),
            "doctor_ID": doctor_ID
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/update-doctor-info/<doctor_ID>", methods=["PUT", "OPTIONS"])
@cross_origin()
def update_doctor_info_controller(doctor_ID):
    """Update doctor information"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        update_data = request.get_json()
        if not update_data:
            raise ValueError("No data provided for update")
        
        updated_doctor = update_doctor(doctor_ID, update_data)
        
        logger.info(f"Doctor info updated successfully: {doctor_ID}")
        return jsonify({
            "message": "Doctor information updated successfully",
            "doctor": mongo_to_dict(updated_doctor)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/update-doctor-password/<doctor_ID>", methods=["PUT", "OPTIONS"])
@cross_origin()
def update_password_controller(doctor_ID):
    """Update doctor password"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        data = request.get_json()
        old_password = data.get("old_password")
        new_password = data.get("new_password")
        
        if not all([old_password, new_password]):
            raise ValueError("Both old and new passwords are required")
        
        if len(new_password) < 8:
            raise ValueError("New password must be at least 8 characters long")
        
        success = update_doctor_password(doctor_ID, old_password, new_password)
        
        if success:
            logger.info(f"Password updated successfully for doctor: {doctor_ID}")
            return jsonify({"message": "Password updated successfully"}), 200
        else:
            raise ValueError("Failed to update password")
            
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/delete-doctor/<doctor_ID>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_doctor_controller(doctor_ID):
    """Delete doctor account"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        delete_doctor(doctor_ID)
        
        logger.info(f"Doctor deleted successfully: {doctor_ID}")
        return jsonify({"message": "Doctor deleted successfully"}), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# PROFILE PICTURE ENDPOINTS
# ================================

@doctor_blueprint.route("/upload-profile-picture/<doctor_ID>", methods=["POST", "OPTIONS"])
@cross_origin()
def upload_profile_picture(doctor_ID):
    """Upload doctor profile picture only"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if "file" not in request.files:
            raise ValueError("No file provided")
        
        file = request.files["file"]
        
        # Validate file for profile picture
        filename = _validate_file_upload(file, ALLOWED_IMAGE_EXTENSIONS, max_size_mb=5)
        
        # Save file to GridFS
        content_type = file.content_type or "image/jpeg"
        file_id = save_file_to_gridfs(file, content_type)
        
        # Get doctor and update profile picture
        doctor = get_doctor_by_id(doctor_ID)
        
        if not doctor.personal_info:
            raise ValueError("Doctor personal info not found")
        
        doctor.personal_info.profile_picture = ObjectId(file_id)
        doctor.save()
        
        logger.info(f"Profile picture uploaded successfully for doctor {doctor_ID}: {file_id}")
        return jsonify({
            "message": "Profile picture uploaded successfully",
            "file_id": file_id,
            "filename": filename
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/get-profile-picture/<doctor_ID>", methods=["GET", "OPTIONS"])
@cross_origin()
def get_profile_picture(doctor_ID):
    """Get doctor profile picture"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        doctor = get_doctor_by_id(doctor_ID)
        
        if not doctor.personal_info or not doctor.personal_info.profile_picture:
            raise ValueError("No profile picture found")
        
        filename, content, content_type = get_file_content_from_gridfs(
            str(doctor.personal_info.profile_picture)
        )
        
        return Response(content, mimetype=content_type or 'image/jpeg')
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# PATIENT MANAGEMENT ENDPOINTS
# ================================

@doctor_blueprint.route("/<doctor_ID>/patients", methods=["GET", "OPTIONS"])
@cross_origin()
def list_patients_for_doctor(doctor_ID):
    """List all patients for a doctor"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        patients = list_patients_by_doctor(doctor_ID)
        
        # Import the patient conversion function
        from controller.patient_controller import mongo_to_dict_patient
        patients_list = [mongo_to_dict_patient(p) for p in patients]
        
        logger.info(f"Listed {len(patients_list)} patients for doctor {doctor_ID}")
        return jsonify({
            "patients": patients_list,
            "doctor_ID": doctor_ID,
            "count": len(patients_list)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/<doctor_ID>/patients", methods=["POST", "OPTIONS"])
@cross_origin()
def add_patient_for_doctor(doctor_ID):
    """Create a new patient for a doctor"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        patient_data = request.get_json() or {}
        
        # Add doctor ID to patient data
        patient_data["doctorID"] = doctor_ID
        
        patient = create_patient_for_doctor(patient_data)
        
        logger.info(f"Patient created for doctor {doctor_ID}: {patient.patientID}")
        return jsonify({
            "message": "Patient created successfully",
            "patientID": patient.patientID,
            "doctorID": patient.doctorID,
        }), 201
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# SESSION SCHEDULING ENDPOINTS
# ================================

@doctor_blueprint.route("/schedule-session/<doctor_ID>", methods=["POST", "OPTIONS"])
@cross_origin()
def schedule_session_controller(doctor_ID):
    """Schedule a new session for a doctor"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        
        data = request.get_json() or {}
        patient_id = data.get("patientID")
        when_iso = data.get("datetime")
        notes = data.get("notes", "")
        
        if not all([patient_id, when_iso]):
            raise ValueError("patientID and datetime are required")
        
        # Validate datetime format
        try:
            datetime.datetime.fromisoformat(when_iso.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
        
        # Use the repository function
        new_session = schedule_session_for_doctor(
            doctor_ID, int(patient_id), when_iso, notes
        )
        
        session_dict = new_session.to_mongo().to_dict()
        session_dict["datetime"] = new_session.datetime.isoformat()
        
        logger.info(f"Session scheduled for doctor {doctor_ID}: patient {patient_id}")
        return jsonify({
            "message": "Session scheduled successfully",
            "scheduledSession": session_dict,
            "doctor_ID": doctor_ID
        }), 201
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/schedule-session/<doctor_ID>/sessions", methods=["GET", "OPTIONS"])
@cross_origin()
def get_scheduled_sessions(doctor_ID):
    """Get all scheduled sessions for a doctor"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        sessions = get_scheduled_sessions_for_doctor(doctor_ID)
        result = []
        
        for session in sessions:
            session_dict = session.to_mongo().to_dict()
            session_dict["datetime"] = session.datetime.isoformat()
            
            # Fetch patient information
            try:
                patient = get_patient_by_id(session.patientID)
                
                if patient and patient.personalInfo:
                    personal_info = patient.personalInfo
                    contact_info = getattr(personal_info, 'contact_information', None)
                    
                    session_dict["patientName"] = getattr(personal_info, 'full_name', None) or f"Patient {session.patientID}"
                    session_dict["patientEmail"] = getattr(contact_info, 'email', None) if contact_info else None
                else:
                    session_dict["patientName"] = f"Patient {session.patientID}"
                    session_dict["patientEmail"] = None
                    
            except Exception as e:
                logger.warning(f"Error fetching patient info for ID {session.patientID}: {e}")
                session_dict["patientName"] = f"Patient {session.patientID}"
                session_dict["patientEmail"] = None
            
            result.append(session_dict)
        
        logger.info(f"Retrieved {len(result)} scheduled sessions for doctor {doctor_ID}")
        return jsonify({
            "sessions": result,
            "doctor_ID": doctor_ID,
            "count": len(result)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# DEBUG ENDPOINTS
# ================================

@doctor_blueprint.route("/debug/doctor-info/<doctor_ID>", methods=["GET"])
@cross_origin()
def debug_doctor_info(doctor_ID):
    """Debug endpoint to get comprehensive doctor information"""
    try:
        from model.doctor_model import Doctor
        
        # Direct query
        doctor = Doctor.objects(doctor_ID=doctor_ID).first()
        
        if not doctor:
            return jsonify({
                "found": False,
                "searched_for": doctor_ID,
                "message": f"Doctor {doctor_ID} not found"
            }), 404
        
        return jsonify({
            "found": True,
            "doctor_id": doctor.doctor_ID,
            "email": doctor.personal_info.email if doctor.personal_info else None,
            "verified": doctor.email_verified,
            "has_scheduled_sessions": bool(doctor.scheduledSessions),
            "session_count": len(doctor.scheduledSessions or []),
            "patient_count": len(doctor.patientIDs or []),
            "workplace": doctor.workplace,
            "license_number": doctor.license_number
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

# ================================
# UTILITY ENDPOINTS
# ================================

@doctor_blueprint.route("/check-token/<token>", methods=["GET", "OPTIONS"])
@cross_origin()
def check_verification_token(token):
    """Check verification token validity without verifying"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        token_status = check_token_validity(token)
        
        return jsonify({
            "valid": token_status["valid"],
            "reason": token_status["reason"],
            "expired": token_status.get("expired", False),
            "doctor_email": token_status.get("doctor_email")
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@doctor_blueprint.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Basic connectivity test
        doctor_count = Doctor.objects.count()
        
        return jsonify({
            "status": "healthy",
            "service": "doctor_controller",
            "doctor_count": doctor_count,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "doctor_controller",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 500

# ================================
# DOCTOR ANALYTICS ENDPOINTS
# ================================

@doctor_blueprint.route("/<doctor_ID>/analytics/patients", methods=["GET", "OPTIONS"])
@cross_origin()
def get_doctor_patient_analytics(doctor_ID):
    """Get analytics for doctor's patients"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        patients = list_patients_by_doctor(doctor_ID)
        
        # Basic analytics
        analytics = {
            "total_patients": len(patients),
            "patients_with_sessions": 0,
            "total_sessions": 0,
            "recent_activity": [],
            "patient_demographics": {
                "by_gender": {},
                "age_groups": {}
            }
        }
        
        for patient in patients:
            # Count sessions
            if hasattr(patient, 'sessions') and patient.sessions:
                analytics["patients_with_sessions"] += 1
                analytics["total_sessions"] += len(patient.sessions)
                
                # Add recent activity (last 5 sessions)
                for session in patient.sessions[-5:]:
                    if hasattr(session, 'date'):
                        analytics["recent_activity"].append({
                            "patient_id": patient.patientID,
                            "patient_name": patient.personalInfo.full_name if patient.personalInfo else f"Patient {patient.patientID}",
                            "session_date": session.date.isoformat() if hasattr(session.date, 'isoformat') else str(session.date),
                            "session_id": getattr(session, 'session_id', 'N/A')
                        })
            
            # Demographics (if available)
            if patient.personalInfo:
                # Gender distribution
                gender = getattr(patient.personalInfo, 'gender', 'Unknown')
                analytics["patient_demographics"]["by_gender"][gender] = analytics["patient_demographics"]["by_gender"].get(gender, 0) + 1
                
                # Age groups (if date_of_birth available)
                dob = getattr(patient.personalInfo, 'date_of_birth', None)
                if dob:
                    try:
                        from datetime import datetime
                        if isinstance(dob, str):
                            birth_date = datetime.strptime(dob, "%Y-%m-%d")
                        else:
                            birth_date = dob
                        
                        age = (datetime.now() - birth_date).days // 365
                        
                        if age < 18:
                            age_group = "Under 18"
                        elif age < 30:
                            age_group = "18-29"
                        elif age < 50:
                            age_group = "30-49"
                        elif age < 65:
                            age_group = "50-64"
                        else:
                            age_group = "65+"
                        
                        analytics["patient_demographics"]["age_groups"][age_group] = analytics["patient_demographics"]["age_groups"].get(age_group, 0) + 1
                    except:
                        pass  # Skip if date parsing fails
        
        # Sort recent activity by date (most recent first)
        analytics["recent_activity"].sort(key=lambda x: x["session_date"], reverse=True)
        analytics["recent_activity"] = analytics["recent_activity"][:10]  # Limit to 10 most recent
        
        # Calculate additional metrics
        analytics["session_statistics"] = {
            "average_sessions_per_patient": analytics["total_sessions"] / analytics["total_patients"] if analytics["total_patients"] > 0 else 0,
            "patients_without_sessions": analytics["total_patients"] - analytics["patients_with_sessions"],
            "engagement_rate": (analytics["patients_with_sessions"] / analytics["total_patients"] * 100) if analytics["total_patients"] > 0 else 0
        }
        
        return jsonify({
            "doctor_ID": doctor_ID,
            "analytics": analytics
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({"error": f"Failed to generate analytics: {str(e)}"}), 500