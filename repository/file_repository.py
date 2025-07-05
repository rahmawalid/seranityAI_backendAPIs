"""
File Repository - Handles file operations with GridFS
Fixed to align with patient, doctor, and report repository patterns
"""

import os
import tempfile
from bson import ObjectId
from typing import Optional, Tuple, Dict, Any
import logging
import mimetypes
from pathlib import Path
import gridfs

from config import fs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported file types and their extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
SUPPORTED_DOCUMENT_EXTENSIONS = ['.pdf', '.doc', '.docx', '.txt']
SUPPORTED_SPREADSHEET_EXTENSIONS = ['.xlsx', '.xls', '.csv']
SUPPORTED_AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.ogg']
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

ALL_SUPPORTED_EXTENSIONS = (
    SUPPORTED_IMAGE_EXTENSIONS + 
    SUPPORTED_DOCUMENT_EXTENSIONS + 
    SUPPORTED_SPREADSHEET_EXTENSIONS + 
    SUPPORTED_AUDIO_EXTENSIONS + 
    SUPPORTED_VIDEO_EXTENSIONS
)

# ------------------------------
# Helper Functions
# ------------------------------

def _handle_gridfs_operation(operation_func, error_message: str):
    """Helper to handle GridFS operations with proper error handling"""
    try:
        return operation_func()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        raise ValueError(f"{error_message}: {str(e)}")

def _validate_file_id(file_id: str) -> ObjectId:
    """Validate and convert file ID to ObjectId"""
    if not file_id:
        raise ValueError("File ID is required")
    
    try:
        # Handle different file_id formats
        if isinstance(file_id, dict) and "$oid" in file_id:
            file_id = file_id["$oid"]
        
        return ObjectId(file_id)
    except Exception as e:
        raise ValueError(f"Invalid file ID format: {file_id}")

def _validate_file_path(file_path: str, check_exists: bool = True) -> str:
    """Validate file path"""
    if not file_path:
        raise ValueError("File path is required")
    
    if check_exists and not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    
    return os.path.abspath(file_path)

def _ensure_directory_exists(file_path: str):
    """Ensure target directory exists"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create directory {directory}: {str(e)}")

def _get_content_type(filename: str) -> str:
    """Get content type from filename"""
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or 'application/octet-stream'

def _validate_file_extension(filename: str, allowed_extensions: list = None) -> str:
    """Validate file extension"""
    if not filename:
        raise ValueError("Filename is required")
    
    _, ext = os.path.splitext(filename.lower())
    
    if allowed_extensions is None:
        allowed_extensions = ALL_SUPPORTED_EXTENSIONS
    
    if ext not in [e.lower() for e in allowed_extensions]:
        raise ValueError(f"Unsupported file extension: {ext}. Allowed: {allowed_extensions}")
    
    return ext

# ------------------------------
# Core File Operations
# ------------------------------

def save_file_to_gridfs(file_path: str, filename: Optional[str] = None, content_type: Optional[str] = None) -> str:
    """
    Save file to GridFS with proper validation and error handling
    
    Args:
        file_path: Path to the file to save
        filename: Optional filename (uses file_path basename if None)
        content_type: Optional content type (auto-detected if None)
        
    Returns:
        str: GridFS file ID
    """
    try:
        # Validate inputs
        file_path = _validate_file_path(file_path)
        
        if filename is None:
            filename = os.path.basename(file_path)
        
        if content_type is None:
            content_type = _get_content_type(filename)
        
        # Validate file extension
        _validate_file_extension(filename)
        
        # Save to GridFS
        def save_operation():
            with open(file_path, "rb") as f:
                return fs.put(f, filename=filename, content_type=content_type)
        
        file_id = _handle_gridfs_operation(
            save_operation,
            f"Failed to save file {filename} to GridFS"
        )
        
        logger.info(f"Successfully saved file {filename} to GridFS with ID {file_id}")
        return str(file_id)
        
    except Exception as e:
        raise ValueError(f"Failed to save file to GridFS: {str(e)}")
    
def get_gridfs_file_object(file_id: str):
    """
    Get GridFS file object for reading
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        GridOut: GridFS file object
    """
    try:
        object_id = _validate_file_id(file_id)
        
        def get_file_operation():
            return fs.get(object_id)
        
        return _handle_gridfs_operation(
            get_file_operation,
            f"Failed to retrieve GridFS object {file_id}"
        )
        
    except Exception as e:
        raise ValueError(f"Failed to get GridFS object: {str(e)}")

def get_file_from_gridfs(file_id: str, target_path: str, preserve_extension: bool = True) -> str:
    """
    Retrieve file from GridFS and save to target path
    
    Args:
        file_id: GridFS file ID
        target_path: Target path to save the file
        preserve_extension: Whether to preserve original file extension
        
    Returns:
        str: Actual output path (may include extension)
    """
    try:
        # Validate inputs
        object_id = _validate_file_id(file_id)
        
        # Get file from GridFS
        def get_file_operation():
            return fs.get(object_id)
        
        grid_out = _handle_gridfs_operation(
            get_file_operation,
            f"Failed to retrieve file {file_id} from GridFS"
        )
        
        # Determine output path
        output_path = target_path
        if preserve_extension:
            original_name = grid_out.filename or ""
            _, ext = os.path.splitext(original_name)
            
            if ext and not output_path.lower().endswith(ext.lower()):
                output_path += ext
        
        # Ensure target directory exists
        _ensure_directory_exists(output_path)
        
        # Write file
        try:
            with open(output_path, "wb") as f:
                f.write(grid_out.read())
        except Exception as e:
            raise ValueError(f"Failed to write file to {output_path}: {str(e)}")
        
        logger.info(f"Successfully retrieved file {file_id} to {output_path}")
        return output_path
        
    except Exception as e:
        raise ValueError(f"Failed to get file from GridFS: {str(e)}")

def get_file_content_from_gridfs(file_id: str) -> Tuple[str, bytes, str]:
    """
    Get file content from GridFS without saving to disk
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        Tuple[str, bytes, str]: (filename, content, content_type)
    """
    try:
        object_id = _validate_file_id(file_id)
        
        def get_file_operation():
            return fs.get(object_id)
        
        grid_out = _handle_gridfs_operation(
            get_file_operation,
            f"Failed to retrieve file content for {file_id}"
        )
        
        filename = grid_out.filename or f"file_{file_id}"
        content = grid_out.read()
        content_type = getattr(grid_out, 'content_type', None) or _get_content_type(filename)
        
        return filename, content, content_type
        
    except Exception as e:
        raise ValueError(f"Failed to get file content from GridFS: {str(e)}")

def delete_file_from_gridfs(file_id: str) -> bool:
    """
    Delete file from GridFS
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        bool: True if deleted successfully
    """
    try:
        object_id = _validate_file_id(file_id)
        
        def delete_operation():
            fs.delete(object_id)
            return True
        
        result = _handle_gridfs_operation(
            delete_operation,
            f"Failed to delete file {file_id} from GridFS"
        )
        
        logger.info(f"Successfully deleted file {file_id} from GridFS")
        return result
        
    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {e}")
        return False

def check_file_exists_in_gridfs(file_id: str) -> bool:
    """
    Check if file exists in GridFS
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        bool: True if file exists
    """
    try:
        object_id = _validate_file_id(file_id)
        
        def check_operation():
            fs.get(object_id)
            return True
        
        return _handle_gridfs_operation(
            check_operation,
            f"File {file_id} not found in GridFS"
        )
        
    except Exception:
        return False

def get_file_metadata_from_gridfs(file_id: str) -> Dict[str, Any]:
    """
    Get file metadata from GridFS
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        Dict[str, Any]: File metadata
    """
    try:
        object_id = _validate_file_id(file_id)
        
        def get_metadata_operation():
            return fs.get(object_id)
        
        grid_out = _handle_gridfs_operation(
            get_metadata_operation,
            f"Failed to get metadata for file {file_id}"
        )
        
        return {
            "file_id": str(grid_out._id),
            "filename": grid_out.filename,
            "content_type": getattr(grid_out, 'content_type', None),
            "length": grid_out.length,
            "upload_date": grid_out.upload_date,
            "chunk_size": grid_out.chunk_size,
            "md5": getattr(grid_out, 'md5', None)
        }
        
    except Exception as e:
        raise ValueError(f"Failed to get file metadata: {str(e)}")

# ------------------------------
# Specialized File Type Operations
# ------------------------------

def save_excel_to_gridfs(file_path: str, filename: Optional[str] = None) -> str:
    """
    Save Excel file to GridFS with validation
    
    Args:
        file_path: Path to Excel file
        filename: Optional filename
        
    Returns:
        str: GridFS file ID
    """
    try:
        # Validate file path
        file_path = _validate_file_path(file_path)
        
        if filename is None:
            filename = os.path.basename(file_path)
        
        # Validate Excel extension
        _validate_file_extension(filename, SUPPORTED_SPREADSHEET_EXTENSIONS)
        
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if filename.lower().endswith('.xls'):
            content_type = "application/vnd.ms-excel"
        elif filename.lower().endswith('.csv'):
            content_type = "text/csv"
        
        return save_file_to_gridfs(file_path, filename, content_type)
        
    except Exception as e:
        raise ValueError(f"Failed to save Excel file: {str(e)}")

def save_image_to_gridfs(file_path: str, filename: Optional[str] = None) -> str:
    """
    Save image file to GridFS with validation
    
    Args:
        file_path: Path to image file
        filename: Optional filename
        
    Returns:
        str: GridFS file ID
    """
    try:
        # Validate file path
        file_path = _validate_file_path(file_path)
        
        if filename is None:
            filename = os.path.basename(file_path)
        
        # Validate image extension
        _validate_file_extension(filename, SUPPORTED_IMAGE_EXTENSIONS)
        
        # Get appropriate content type
        content_type = _get_content_type(filename)
        if not content_type.startswith('image/'):
            # Default to image/jpeg for unknown image types
            content_type = 'image/jpeg'
        
        return save_file_to_gridfs(file_path, filename, content_type)
        
    except Exception as e:
        raise ValueError(f"Failed to save image file: {str(e)}")

def save_pdf_to_gridfs(file_path: str, filename: Optional[str] = None) -> str:
    """
    Save PDF file to GridFS with validation
    
    Args:
        file_path: Path to PDF file
        filename: Optional filename
        
    Returns:
        str: GridFS file ID
    """
    try:
        # Validate file path
        file_path = _validate_file_path(file_path)
        
        if filename is None:
            filename = os.path.basename(file_path)
        
        # Validate PDF extension
        _validate_file_extension(filename, ['.pdf'])
        
        return save_file_to_gridfs(file_path, filename, "application/pdf")
        
    except Exception as e:
        raise ValueError(f"Failed to save PDF file: {str(e)}")

def save_audio_to_gridfs(file_path: str, filename: Optional[str] = None) -> str:
    """
    Save audio file to GridFS with validation
    
    Args:
        file_path: Path to audio file
        filename: Optional filename
        
    Returns:
        str: GridFS file ID
    """
    try:
        # Validate file path
        file_path = _validate_file_path(file_path)
        
        if filename is None:
            filename = os.path.basename(file_path)
        
        # Validate audio extension
        _validate_file_extension(filename, SUPPORTED_AUDIO_EXTENSIONS)
        
        # Get appropriate content type
        content_type = _get_content_type(filename)
        if not content_type.startswith('audio/'):
            content_type = 'audio/mpeg'  # Default to MP3
        
        return save_file_to_gridfs(file_path, filename, content_type)
        
    except Exception as e:
        raise ValueError(f"Failed to save audio file: {str(e)}")

def save_video_to_gridfs(file_path: str, filename: Optional[str] = None) -> str:
    """
    Save video file to GridFS with validation
    
    Args:
        file_path: Path to video file
        filename: Optional filename
        
    Returns:
        str: GridFS file ID
    """
    try:
        # Validate file path
        file_path = _validate_file_path(file_path)
        
        if filename is None:
            filename = os.path.basename(file_path)
        
        # Validate video extension
        _validate_file_extension(filename, SUPPORTED_VIDEO_EXTENSIONS)
        
        # Get appropriate content type
        content_type = _get_content_type(filename)
        if not content_type.startswith('video/'):
            content_type = 'video/mp4'  # Default to MP4
        
        return save_file_to_gridfs(file_path, filename, content_type)
        
    except Exception as e:
        raise ValueError(f"Failed to save video file: {str(e)}")

# ------------------------------
# Temporary File Operations
# ------------------------------

def save_to_temp_file(file_id: str, suffix: Optional[str] = None) -> str:
    """
    Save GridFS file to temporary file
    
    Args:
        file_id: GridFS file ID
        suffix: Optional file suffix/extension
        
    Returns:
        str: Path to temporary file
    """
    try:
        # Get file metadata to determine extension if not provided
        if suffix is None:
            metadata = get_file_metadata_from_gridfs(file_id)
            filename = metadata.get('filename', '')
            if filename:
                _, ext = os.path.splitext(filename)
                suffix = ext
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
        
        # Save file to temporary location
        return get_file_from_gridfs(file_id, temp_path, preserve_extension=False)
        
    except Exception as e:
        raise ValueError(f"Failed to save file to temporary location: {str(e)}")

# ------------------------------
# Utility Functions
# ------------------------------

def validate_gridfs_connection() -> bool:
    """
    Validate GridFS connection
    
    Returns:
        bool: True if connection is working
    """
    try:
        # Try to list files (limited to 1 for quick check)
        list(fs.find().limit(1))
        return True
    except Exception as e:
        logger.error(f"GridFS connection validation failed: {e}")
        return False

def get_supported_extensions() -> Dict[str, list]:
    """
    Get all supported file extensions by category
    
    Returns:
        Dict[str, list]: Dictionary of supported extensions by type
    """
    return {
        "images": SUPPORTED_IMAGE_EXTENSIONS,
        "documents": SUPPORTED_DOCUMENT_EXTENSIONS,
        "spreadsheets": SUPPORTED_SPREADSHEET_EXTENSIONS,
        "audio": SUPPORTED_AUDIO_EXTENSIONS,
        "video": SUPPORTED_VIDEO_EXTENSIONS,
        "all": ALL_SUPPORTED_EXTENSIONS
    }

def get_file_size_in_gridfs(file_id: str) -> int:
    """
    Get file size from GridFS
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        int: File size in bytes
    """
    try:
        metadata = get_file_metadata_from_gridfs(file_id)
        return metadata.get('length', 0)
    except Exception as e:
        raise ValueError(f"Failed to get file size: {str(e)}")

def list_files_in_gridfs(limit: int = 100, skip: int = 0) -> list:
    """
    List files in GridFS with pagination
    
    Args:
        limit: Maximum number of files to return
        skip: Number of files to skip
        
    Returns:
        list: List of file metadata dictionaries
    """
    try:
        files = []
        for grid_out in fs.find().skip(skip).limit(limit):
            files.append({
                "file_id": str(grid_out._id),
                "filename": grid_out.filename,
                "length": grid_out.length,
                "upload_date": grid_out.upload_date,
                "content_type": getattr(grid_out, 'content_type', None)
            })
        return files
    except Exception as e:
        raise ValueError(f"Failed to list files: {str(e)}")