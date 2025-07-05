"""
Doctor Notes Controller Layer - API Endpoints Layer
Handles all HTTP requests for doctor notes functionality
"""

import os
import tempfile
import traceback
from flask import Blueprint, request, jsonify, make_response, send_file
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from io import BytesIO
from bson import ObjectId

from services.doctor_notes_service import DoctorNotesService
from config import fs


# Create Blueprint
doctor_notes_bp = Blueprint("doctor_notes", __name__)

# Initialize Service
doctor_notes_service = DoctorNotesService()


# ================================
# CORS HELPER
# ================================

def _cors_preflight():
    """Handle CORS preflight requests"""
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# ================================
# DOCTOR NOTES UPLOAD ENDPOINTS
# ================================

@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/doctor-notes/upload", methods=["OPTIONS", "POST"])
@cross_origin()
def upload_doctor_notes(patient_id, session_id):
    """
    Upload doctor notes images for a specific patient session
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Request:
        files: Multiple image files (doctor_notes[])
        
    Returns:
        JSON response with upload results
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Check if files are present
        if 'doctor_notes' not in request.files:
            return jsonify({
                "success": False,
                "error": "No doctor_notes files found in request"
            }), 400
        
        files = request.files.getlist('doctor_notes')
        
        if not files or len(files) == 0:
            return jsonify({
                "success": False,
                "error": "No files selected"
            }), 400
        
        # Check for empty filenames
        files = [f for f in files if f.filename != '']
        if not files:
            return jsonify({
                "success": False,
                "error": "All files have empty filenames"
            }), 400
        
        print(f"📤 Uploading {len(files)} doctor notes for patient {patient_id}, session {session_id}")
        
        # Use service to handle upload
        result = doctor_notes_service.upload_doctor_notes(patient_id, session_id, files)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "message": result["message"],
                "data": {
                    "uploaded_files": len(result["file_ids"]),
                    "file_ids": result["file_ids"],
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "validation_summary": {
                        "total_files": result["validation_results"]["total_files"],
                        "valid_files": len(result["validation_results"]["valid_files"]),
                        "invalid_files": len(result["validation_results"]["invalid_files"])
                    }
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"],
                "validation_results": result.get("validation_results")
            }), 400
            
    except Exception as e:
        print(f"❌ Error uploading doctor notes: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error during upload",
            "details": str(e)
        }), 500


@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/doctor-notes/upload-single", methods=["OPTIONS", "POST"])
@cross_origin()
def upload_single_doctor_note(patient_id, session_id):
    """
    Upload a single doctor note image
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Request:
        file: Single image file (doctor_note)
        
    Returns:
        JSON response with upload result
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if 'doctor_note' not in request.files:
            return jsonify({
                "success": False,
                "error": "No doctor_note file found in request"
            }), 400
        
        file = request.files['doctor_note']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        print(f"📤 Uploading single doctor note for patient {patient_id}, session {session_id}")
        
        # Use service to handle upload
        result = doctor_notes_service.upload_doctor_notes(patient_id, session_id, [file])
        
        if result["success"]:
            return jsonify({
                "success": True,
                "message": "Doctor note uploaded successfully",
                "data": {
                    "file_id": result["file_ids"][0],
                    "filename": file.filename,
                    "patient_id": patient_id,
                    "session_id": session_id
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 400
            
    except Exception as e:
        print(f"❌ Error uploading single doctor note: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during upload",
            "details": str(e)
        }), 500


# ================================
# DOCTOR NOTES RETRIEVAL ENDPOINTS
# ================================

@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/doctor-notes", methods=["GET", "OPTIONS"])
@cross_origin()
def get_doctor_notes(patient_id, session_id):
    """
    Get all doctor notes for a specific patient session
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Returns:
        JSON response with doctor notes information
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"📥 Retrieving doctor notes for patient {patient_id}, session {session_id}")
        
        # Use service to get notes information
        result = doctor_notes_service.get_doctor_notes_for_session(patient_id, session_id)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "data": {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "notes_count": result["notes_count"],
                    "has_notes": result["has_notes"],
                    "notes": result["notes_info"]
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 404
            
    except Exception as e:
        print(f"❌ Error retrieving doctor notes: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during retrieval",
            "details": str(e)
        }), 500


@doctor_notes_bp.route("/patient/<patient_id>/doctor-notes/summary", methods=["GET", "OPTIONS"])
@cross_origin()
def get_patient_doctor_notes_summary(patient_id):
    """
    Get summary of all doctor notes for a patient across all sessions
    
    Args:
        patient_id: Patient ID
        
    Returns:
        JSON response with summary statistics
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"📊 Retrieving doctor notes summary for patient {patient_id}")
        
        # Use service to get statistics
        result = doctor_notes_service.get_patient_doctor_notes_statistics(patient_id)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "data": result
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 404
            
    except Exception as e:
        print(f"❌ Error retrieving doctor notes summary: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during summary retrieval",
            "details": str(e)
        }), 500


# ================================
# DOCTOR NOTES DOWNLOAD ENDPOINTS
# ================================

@doctor_notes_bp.route("/doctor-notes/download/<file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def download_doctor_note(file_id):
    """
    Download a specific doctor note by file ID
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        File download response
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"📥 Downloading doctor note: {file_id}")
        
        # Get file data from repository
        filename, file_data, content_type = doctor_notes_service.repository.get_file_data_for_download(file_id)
        
        if file_data is None:
            return jsonify({
                "success": False,
                "error": "Doctor note not found"
            }), 404
        
        # Create file response
        return send_file(
            BytesIO(file_data),
            as_attachment=True,
            attachment_filename=filename or f"doctor_note_{file_id}.jpg",
            mimetype=content_type or "image/jpeg"
        )
        
    except Exception as e:
        print(f"❌ Error downloading doctor note: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during download",
            "details": str(e)
        }), 500


# ================================
# DOCTOR NOTES DELETION ENDPOINTS
# ================================

@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/doctor-notes/<file_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_doctor_note(patient_id, session_id, file_id):
    """
    Delete a specific doctor note from a session
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        file_id: File ID to delete
        
    Returns:
        JSON response with deletion result
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"🗑️ Deleting doctor note {file_id} from patient {patient_id}, session {session_id}")
        
        # Use service to delete note
        result = doctor_notes_service.delete_doctor_note(patient_id, session_id, file_id)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "message": result["message"],
                "data": {
                    "deleted_file_id": file_id,
                    "patient_id": patient_id,
                    "session_id": session_id
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 400
            
    except Exception as e:
        print(f"❌ Error deleting doctor note: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during deletion",
            "details": str(e)
        }), 500


@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/doctor-notes/clear", methods=["DELETE", "OPTIONS"])
@cross_origin()
def clear_all_doctor_notes(patient_id, session_id):
    """
    Clear all doctor notes from a session
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Returns:
        JSON response with clearing result
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"🗑️ Clearing all doctor notes from patient {patient_id}, session {session_id}")
        
        # Get current notes count first
        notes_info = doctor_notes_service.get_doctor_notes_for_session(patient_id, session_id)
        notes_count = notes_info.get("notes_count", 0) if notes_info["success"] else 0
        
        # Clear all notes
        success = doctor_notes_service.repository.clear_all_doctor_notes_from_session(patient_id, session_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Successfully cleared {notes_count} doctor notes",
                "data": {
                    "cleared_notes_count": notes_count,
                    "patient_id": patient_id,
                    "session_id": session_id
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Failed to clear doctor notes"
            }), 400
            
    except Exception as e:
        print(f"❌ Error clearing doctor notes: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during clearing",
            "details": str(e)
        }), 500


# ================================
# ANALYSIS WORKFLOW ENDPOINTS
# ================================

@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/analysis-capabilities", methods=["GET", "OPTIONS"])
@cross_origin()
def get_analysis_capabilities(patient_id, session_id):
    """
    Get analysis capabilities and recommendations for a session
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Returns:
        JSON response with analysis capabilities
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"🔍 Checking analysis capabilities for patient {patient_id}, session {session_id}")
        
        # Use service to get comprehensive capabilities
        result = doctor_notes_service.get_session_capabilities_summary(patient_id, session_id)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "data": result
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 404
            
    except Exception as e:
        print(f"❌ Error checking analysis capabilities: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during capabilities check",
            "details": str(e)
        }), 500


@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/prepare-analysis", methods=["POST", "OPTIONS"])
@cross_origin()
def prepare_analysis_data(patient_id, session_id):
    """
    Prepare analysis data and determine analysis type
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Returns:
        JSON response with prepared analysis data
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"⚙️ Preparing analysis data for patient {patient_id}, session {session_id}")
        
        # Use service to prepare analysis data
        result = doctor_notes_service.prepare_analysis_data(patient_id, session_id)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "message": f"Analysis data prepared for {result['analysis_type']} analysis",
                "data": {
                    "analysis_type": result["analysis_type"],
                    "prompt_type": result["prompt_type"],
                    "prepared_data": result["data"],
                    "patient_id": patient_id,
                    "session_id": session_id
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 400
            
    except Exception as e:
        print(f"❌ Error preparing analysis data: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during analysis preparation",
            "details": str(e)
        }), 500


@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/enhancement-readiness", methods=["GET", "OPTIONS"])
@cross_origin()
def check_enhancement_readiness(patient_id, session_id):
    """
    Check if session is ready for enhanced analysis with doctor notes
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Returns:
        JSON response with enhancement readiness
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        print(f"🔍 Checking enhancement readiness for patient {patient_id}, session {session_id}")
        
        # Use service to check readiness
        result = doctor_notes_service.check_session_enhancement_readiness(patient_id, session_id)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "data": {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "ready_for_enhancement": result["ready_for_enhancement"],
                    "enhancement_status": result["enhancement_status"],
                    "recommendation": result["recommendation"]
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 404
            
    except Exception as e:
        print(f"❌ Error checking enhancement readiness: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during readiness check",
            "details": str(e)
        }), 500


# ================================
# REPORT GENERATION ENDPOINTS
# ================================

# @doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/generate-enhanced-report", methods=["POST", "OPTIONS"])
# @cross_origin()
def generate_enhanced_report(patient_id, session_id):
    """
    Generate enhanced report using doctor notes analysis
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Request JSON:
        {
            "analysis_type": "comprehensive_with_notes" | "speech_with_notes" | "auto",
            "include_images": true|false (optional, default: true)
        }
        
    Returns:
        JSON response with generated report information
    """
    # if request.method == "OPTIONS":
    #     return _cors_preflight()
    
    try:
        # Get request data
        # request_data = request.get_json() or {}
        analysis_type ="auto"
        include_images = True
        
        print(f"📊 Generating enhanced report for patient {patient_id}, session {session_id} (type: {analysis_type})")
        
        # Check if session has doctor notes
        enhancement_readiness = doctor_notes_service.check_session_enhancement_readiness(patient_id, session_id)
        if not enhancement_readiness["ready_for_enhancement"]:
            return {
                "success": False,
                "error": "Session does not have doctor notes for enhanced analysis",
                "recommendation": "Upload doctor notes first to enable enhanced analysis"
            }, 400
        
        # Generate the enhanced report using the service
        result = doctor_notes_service.generate_enhanced_report_with_images(patient_id, session_id)
        print("here after reportttttttt")
        if result["success"]:
            return {
                "success": True,
                "message": "Enhanced report with doctor notes generated successfully",
                "data": {
                    "report_id": result["report_id"],
                    "analysis_type": result["analysis_type"],
                    "doctor_notes_count": result["doctor_notes_count"],
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "prompt_used": result["prompt_used"],
                    "images_included": result["images_included"],
                    "features_included": {
                        "doctor_notes": True,
                        "fer_analysis": "comprehensive" in result["analysis_type"],
                        "speech_analysis": True,
                        "mismatch_analysis": "comprehensive" in result["analysis_type"]
                    }
                }
            }, 200
        else:
            return {
                "success": False,
                "error": "Failed to generate enhanced report",
                "details": result.get("error", "Unknown error")
            }, 400
        
    except Exception as e:
        print(f"❌ Error generating enhanced report: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error during enhanced report generation",
            "details": str(e)
        }), 500

# ================================
# UTILITY ENDPOINTS
# ================================

@doctor_notes_bp.route("/doctor-notes/health", methods=["GET", "OPTIONS"])
@cross_origin()
def health_check():
    """
    Health check endpoint for doctor notes service
    
    Returns:
        JSON response with service health status
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Test GridFS connection
        fs.list()
        
        return jsonify({
            "success": True,
            "service": "doctor_notes",
            "status": "healthy",
            "features": {
                "upload_support": True,
                "ai_analysis": True,
                "image_processing": True,
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
                "max_file_size_mb": 10
            },
            "endpoints": {
                "upload": "/patient/<patient_id>/session/<session_id>/doctor-notes/upload",
                "retrieve": "/patient/<patient_id>/session/<session_id>/doctor-notes",
                "download": "/doctor-notes/download/<file_id>",
                "delete": "/patient/<patient_id>/session/<session_id>/doctor-notes/<file_id>",
                "analysis": "/patient/<patient_id>/session/<session_id>/analysis-capabilities"
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "service": "doctor_notes",
            "status": "unhealthy",
            "error": str(e)
        }), 503


@doctor_notes_bp.route("/doctor-notes/supported-formats", methods=["GET", "OPTIONS"])
@cross_origin()
def get_supported_formats():
    """
    Get list of supported file formats and upload requirements
    
    Returns:
        JSON response with format information
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    return jsonify({
        "success": True,
        "supported_formats": {
            "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
            "max_file_size_mb": 10,
            "max_files_per_upload": 10,
            "recommended_formats": [".jpg", ".png"]
        },
        "upload_requirements": {
            "field_name": "doctor_notes",
            "field_name_single": "doctor_note",
            "content_type": "multipart/form-data",
            "validation": {
                "required_extension": True,
                "max_file_size": "10MB",
                "image_validation": True
            }
        },
        "recommendations": {
            "image_quality": "High resolution for better AI analysis",
            "file_format": "JPEG or PNG for best compatibility",
            "image_orientation": "Portrait or landscape both supported",
            "handwriting": "Clear, legible handwriting preferred"
        }
    }), 200


# ================================
# ERROR HANDLERS
# ================================

@doctor_notes_bp.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "error": "File too large",
        "message": "File size exceeds maximum allowed size (10MB)",
        "code": 413
    }), 413


@doctor_notes_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request error"""
    return jsonify({
        "success": False,
        "error": "Bad request",
        "message": "Invalid request format or parameters",
        "code": 400
    }), 400


@doctor_notes_bp.errorhandler(404)
def not_found(error):
    """Handle not found error"""
    return jsonify({
        "success": False,
        "error": "Not found",
        "message": "Requested resource not found",
        "code": 404
    }), 404


@doctor_notes_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "code": 500
    }), 500



@doctor_notes_bp.route("/patient/<patient_id>/session/<int:session_id>/integrate-with-existing-report", methods=["POST", "OPTIONS"])
@cross_origin()
def integrate_with_existing_report(patient_id, session_id):
    """
    Integrate doctor notes with existing report generation system
    
    Args:
        patient_id: Patient ID
        session_id: Session ID
        
    Request JSON:
        {
            "report_type": "fer_tov" | "tov_only",
            "existing_report_id": "optional_existing_report_id"
        }
        
    Returns:
        JSON response with integration status
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Get request data
        request_data = request.get_json() or {}
        report_type = request_data.get("report_type", "auto")
        existing_report_id = request_data.get("existing_report_id")
        
        print(f"🔗 Integrating doctor notes with existing report for patient {patient_id}, session {session_id}")
        
        # Check if session has doctor notes
        enhancement_readiness = doctor_notes_service.check_session_enhancement_readiness(patient_id, session_id)
        if not enhancement_readiness["ready_for_enhancement"]:
            return jsonify({
                "success": False,
                "error": "Session does not have doctor notes for integration",
                "recommendation": "Upload doctor notes first to enable integration"
            }), 400
        
        # Prepare integration data
        prepared_data = doctor_notes_service.prepare_analysis_data(patient_id, session_id)
        if not prepared_data["success"]:
            return jsonify({
                "success": False,
                "error": f"Failed to prepare integration data: {prepared_data['error']}"
            }), 400
        
        # Return integration data for existing report system
        return jsonify({
            "success": True,
            "message": "Doctor notes integration data prepared successfully",
            "data": {
                "patient_id": patient_id,
                "session_id": session_id,
                "integration_type": prepared_data["analysis_type"],
                "doctor_notes_data": {
                    "notes_count": prepared_data["data"]["doctor_notes_count"],
                    "has_notes": prepared_data["data"]["has_doctor_notes"],
                    "patient_info": prepared_data["data"]["patient_info"],
                    "session_info": prepared_data["data"]["session_info"]
                },
                "recommended_workflow": {
                    "comprehensive_with_notes": "Use FER + TOV + Doctor Notes analysis",
                    "speech_with_notes": "Use TOV + Doctor Notes analysis",
                    "notes_only": "Use Doctor Notes only analysis"
                }.get(prepared_data["analysis_type"], "Standard analysis"),
                "integration_ready": True
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error integrating with existing report: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error during integration",
            "details": str(e)
        }), 500
    
@doctor_notes_bp.route("/doctor-notes/validate-files", methods=["POST", "OPTIONS"])
@cross_origin()
def validate_files():
    """
    Validate doctor notes files before upload
    
    Returns:
        JSON response with validation results
    """
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Check if files are present
        if 'doctor_notes' not in request.files:
            return jsonify({
                "success": False,
                "error": "No doctor_notes files found in request"
            }), 400
        
        files = request.files.getlist('doctor_notes')
        
        if not files or len(files) == 0:
            return jsonify({
                "success": False,
                "error": "No files selected"
            }), 400
        
        # Use service to validate files
        validation_results = doctor_notes_service.validate_uploaded_files(files)
        
        return jsonify({
            "success": True,
            "message": "File validation completed",
            "data": {
                "total_files": validation_results["total_files"],
                "valid_files": len(validation_results["valid_files"]),
                "invalid_files": len(validation_results["invalid_files"]),
                "validation_details": {
                    "valid_files": [
                        {
                            "index": f["index"],
                            "filename": f["filename"],
                            "status": "valid"
                        }
                        for f in validation_results["valid_files"]
                    ],
                    "invalid_files": validation_results["invalid_files"],
                    "errors": validation_results["errors"]
                }
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error validating files: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error during validation",
            "details": str(e)
        }), 500