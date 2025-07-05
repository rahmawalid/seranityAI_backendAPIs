"""
File Controller - HTTP Request Handler Layer
Handles all HTTP requests for file-related functionality
Enhanced with better error handling and additional endpoints
"""

import os
import tempfile
import logging
from flask import Blueprint, request, jsonify, make_response, send_file, Response
from flask_cors import cross_origin
from bson import ObjectId
from werkzeug.utils import secure_filename
from io import BytesIO

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Repository Layer Imports
from repository.file_repository import (
    get_file_from_gridfs,
    get_file_content_from_gridfs,
    save_file_to_gridfs,
    delete_file_from_gridfs,
    get_file_metadata_from_gridfs,
    check_file_exists_in_gridfs,
    get_supported_extensions,
    validate_gridfs_connection,
    list_files_in_gridfs,
    get_file_size_in_gridfs,
)

from config import fs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
file_bp = Blueprint("file", __name__)

# File type configurations
ALLOWED_EXTENSIONS = {
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'],
    'document': ['.pdf', '.doc', '.docx', '.txt'],
    'spreadsheet': ['.xlsx', '.xls', '.csv'],
    'audio': ['.mp3', '.wav', '.m4a', '.ogg'],
    'video': ['.mp4', '.avi', '.mov', '.mkv']
}

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
    logger.error(f"File controller error: {error}")
    
    if isinstance(error, ValueError):
        return jsonify({"error": "Invalid input", "details": str(error)}), 400
    elif "not found" in str(error).lower():
        return jsonify({"error": "File not found", "details": str(error)}), 404
    else:
        return jsonify({"error": "Internal server error", "details": str(error)}), default_status

def _validate_file_upload(file, allowed_extensions=None, max_size_mb=50):
    """Validate file upload with proper error messages"""
    if not file or file.filename == "":
        raise ValueError("No file provided")
    
    filename = secure_filename(file.filename)
    if not filename:
        raise ValueError("Invalid filename")
    
    # Check extension if specified
    if allowed_extensions:
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

# ================================
# ORIGINAL EXCEL DOWNLOAD ENDPOINT (UNCHANGED LOGIC)
# ================================

@file_bp.route('/file/excel/<file_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_excel_file(file_id):
    """Download Excel File by File ID - Original function with enhanced error handling"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Original logic preserved exactly
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        grid_out = fs.get(ObjectId(file_id))
        tmp.write(grid_out.read())
        tmp.close()

        return send_file(
            tmp.name,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            attachment_filename=f"fer_analysis_{file_id}.xlsx"
        )
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# NEW FILE MANAGEMENT ENDPOINTS
# ================================

@file_bp.route('/file/download/<file_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def download_file(file_id):
    """Generic file download endpoint"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not file_id:
            raise ValueError("File ID is required")
        
        filename, content, content_type = get_file_content_from_gridfs(file_id)
        
        logger.info(f"File downloaded: {filename} ({file_id})")
        return Response(
            content,
            mimetype=content_type or 'application/octet-stream',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Length': str(len(content))
            }
        )
        
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/view/<file_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def view_file(file_id):
    """View file in browser (inline display)"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not file_id:
            raise ValueError("File ID is required")
        
        filename, content, content_type = get_file_content_from_gridfs(file_id)
        
        logger.info(f"File viewed: {filename} ({file_id})")
        return Response(
            content,
            mimetype=content_type or 'application/octet-stream',
            headers={
                'Content-Disposition': f'inline; filename="{filename}"',
                'Content-Length': str(len(content))
            }
        )
        
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/metadata/<file_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_file_metadata(file_id):
    """Get file metadata"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not file_id:
            raise ValueError("File ID is required")
        
        metadata = get_file_metadata_from_gridfs(file_id)
        
        logger.info(f"Metadata retrieved for file: {file_id}")
        return jsonify({
            "message": "File metadata retrieved successfully",
            "file_id": file_id,
            "metadata": metadata
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/upload', methods=['POST', 'OPTIONS'])
@cross_origin()
def upload_file():
    """Generic file upload endpoint"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if 'file' not in request.files:
            raise ValueError("No file provided in request")
        
        file = request.files['file']
        file_type = request.form.get('type', 'generic')  # Optional file type category
        
        # Get allowed extensions based on type
        allowed_extensions = None
        if file_type in ALLOWED_EXTENSIONS:
            allowed_extensions = ALLOWED_EXTENSIONS[file_type]
        
        filename = _validate_file_upload(file, allowed_extensions)
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file.save(tmp_file.name)
            
            # Save to GridFS
            file_id = save_file_to_gridfs(
                tmp_file.name, 
                filename=filename,
                content_type=file.content_type
            )
        
        # Clean up temporary file
        os.unlink(tmp_file.name)
        
        logger.info(f"File uploaded successfully: {filename} -> {file_id}")
        return jsonify({
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": filename,
            "file_type": file_type
        }), 201
        
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/<file_id>', methods=['DELETE', 'OPTIONS'])
@cross_origin()
def delete_file(file_id):
    """Delete file from GridFS"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not file_id:
            raise ValueError("File ID is required")
        
        success = delete_file_from_gridfs(file_id)
        
        if success:
            logger.info(f"File deleted successfully: {file_id}")
            return jsonify({
                "message": "File deleted successfully",
                "file_id": file_id
            }), 200
        else:
            raise ValueError("Failed to delete file")
            
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/exists/<file_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def check_file_exists(file_id):
    """Check if file exists in GridFS"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not file_id:
            raise ValueError("File ID is required")
        
        exists = check_file_exists_in_gridfs(file_id)
        
        return jsonify({
            "file_id": file_id,
            "exists": exists
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/size/<file_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_file_size(file_id):
    """Get file size"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        if not file_id:
            raise ValueError("File ID is required")
        
        size = get_file_size_in_gridfs(file_id)
        
        return jsonify({
            "file_id": file_id,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2)
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# UTILITY ENDPOINTS
# ================================

@file_bp.route('/file/supported-extensions', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_supported_file_extensions():
    """Get list of supported file extensions"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        extensions = get_supported_extensions()
        
        return jsonify({
            "message": "Supported extensions retrieved",
            "extensions": extensions,
            "categories": ALLOWED_EXTENSIONS
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/list', methods=['GET', 'OPTIONS'])
@cross_origin()
def list_files():
    """List files in GridFS with pagination"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Get query parameters
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100 files
        skip = int(request.args.get('skip', 0))
        
        files = list_files_in_gridfs(limit=limit, skip=skip)
        
        return jsonify({
            "message": "Files listed successfully",
            "files": files,
            "count": len(files),
            "limit": limit,
            "skip": skip
        }), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@file_bp.route('/file/health', methods=['GET', 'OPTIONS'])
@cross_origin()
def file_health_check():
    """Health check for file service"""
    if request.method == "OPTIONS":
        return _cors_preflight()
    
    try:
        # Check GridFS connection
        is_connected = validate_gridfs_connection()
        
        if is_connected:
            return jsonify({
                "status": "healthy",
                "service": "file_controller",
                "gridfs_connected": True,
                "supported_extensions": len(get_supported_extensions()["all"])
            }), 200
        else:
            return jsonify({
                "status": "unhealthy",
                "service": "file_controller",
                "gridfs_connected": False,
                "error": "GridFS connection failed"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "file_controller",
            "error": str(e)
        }), 500