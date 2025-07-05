"""
Report Repository - Pure Data Access Layer
Handles only GridFS operations and basic file management for reports
"""

import io
import os
import tempfile
from bson import ObjectId
import gridfs
import pandas as pd
from PIL import Image
from pymongo import MongoClient
from typing import List, Optional, Tuple
import logging
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

# Global temporary files tracking for cleanup
_temp_files = []

def _cleanup_temp_files():
    """Clean up temporary files on exit"""
    global _temp_files
    for temp_file in _temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
    _temp_files.clear()

# Register cleanup function
atexit.register(_cleanup_temp_files)

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

# ------------------------------
# GridFS File Operations
# ------------------------------

def save_pdf_to_gridfs(pdf_stream: io.BytesIO, filename: str) -> str:
    """
    Save PDF bytes to GridFS
    
    Args:
        pdf_stream: PDF content as BytesIO
        filename: Name for the PDF file
        
    Returns:
        str: GridFS file ID
    """
    try:
        def save_operation():
            return fs.put(
                pdf_stream,
                filename=filename,
                contentType="application/pdf",
            )
        
        return str(_handle_gridfs_operation(
            save_operation,
            "Failed to save PDF to GridFS"
        ))
        
    except Exception as e:
        raise ValueError(f"Failed to save PDF to GridFS: {str(e)}")

def get_file_from_gridfs(file_id: str) -> Tuple[str, bytes, str]:
    """
    Retrieve file from GridFS by ID
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        Tuple[str, bytes, str]: (filename, content, content_type)
    """
    try:
        def get_file_operation():
            return fs.get(ObjectId(file_id))
        
        file = _handle_gridfs_operation(
            get_file_operation,
            f"Failed to retrieve file {file_id}"
        )
        
        return file.filename, file.read(), file.content_type
        
    except Exception as e:
        logger.error(f"Error fetching file {file_id}: {e}")
        raise ValueError(f"Failed to retrieve file: {str(e)}")

def get_gridfs_file_object(file_id: str):
    """
    Get GridFS file object for reading
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        GridOut: GridFS file object
    """
    try:
        def get_file_operation():
            return fs.get(ObjectId(file_id))
        
        return _handle_gridfs_operation(
            get_file_operation,
            f"Failed to retrieve GridFS object {file_id}"
        )
        
    except Exception as e:
        raise ValueError(f"Failed to get GridFS object: {str(e)}")

def save_gridout_images_to_tempfiles(photo_gridouts: List) -> List[str]:
    """
    Save GridFS image files to temporary files
    
    Args:
        photo_gridouts: List of GridFS GridOut objects
        
    Returns:
        List[str]: List of temporary file paths
    """
    global _temp_files
    saved_paths = []
    
    try:
        for i, grid_out in enumerate(photo_gridouts):
            if not grid_out:
                logger.warning(f"Skipping empty GridOut object at index {i}")
                continue
                
            try:
                suffix = os.path.splitext(grid_out.filename)[-1] if grid_out.filename else '.png'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(grid_out.read())
                    saved_paths.append(tmp_file.name)
                    _temp_files.append(tmp_file.name)  # Track for cleanup
                    
            except Exception as e:
                logger.error(f"Failed to save GridOut {i} to temp file: {e}")
                continue
                
        return saved_paths
        
    except Exception as e:
        # Cleanup any files created so far on error
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
        raise ValueError(f"Failed to save GridOut images to temp files: {str(e)}")

def load_dataframe_from_gridfs(file_id: str, file_type: Optional[str] = None) -> pd.DataFrame:
    """
    Load DataFrame from GridFS file
    
    Args:
        file_id: GridFS file ID
        file_type: File type ('csv' or 'excel'), auto-detected if None
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        # Handle different file_id formats
        if isinstance(file_id, dict) and "$oid" in file_id:
            file_id = file_id["$oid"]
        
        file_id = ObjectId(file_id)
        
        grid_out = get_gridfs_file_object(str(file_id))
        
        filename = grid_out.filename.lower() if grid_out.filename else ""
        file_data = grid_out.read()
        buffer = io.BytesIO(file_data)
        
        # Auto-detect file type if not provided
        if file_type is None:
            if filename.endswith(".csv"):
                file_type = "csv"
            elif filename.endswith((".xlsx", ".xls")):
                file_type = "excel"
            else:
                raise ValueError(f"Cannot auto-detect file type from filename: {filename}")
        
        # Read the file into a DataFrame
        if file_type == "csv":
            try:
                return pd.read_csv(buffer, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed. Trying ISO-8859-1...")
                buffer.seek(0)
                return pd.read_csv(buffer, encoding="ISO-8859-1")
                
        elif file_type in ["excel", "xlsx", "xls"]:
            return pd.read_excel(buffer)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        raise ValueError(f"Failed to load DataFrame from GridFS: {str(e)}")

def check_file_exists_in_gridfs(file_id: str) -> bool:
    """
    Check if file exists in GridFS
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        if not file_id:
            return False
            
        fs.get(ObjectId(file_id))
        return True
        
    except Exception:
        return False

def get_file_metadata_from_gridfs(file_id: str) -> dict:
    """
    Get file metadata from GridFS
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        dict: File metadata
    """
    try:
        grid_out = get_gridfs_file_object(file_id)
        
        return {
            "filename": grid_out.filename,
            "content_type": grid_out.content_type,
            "length": grid_out.length,
            "upload_date": grid_out.upload_date,
            "file_id": str(grid_out._id)
        }
        
    except Exception as e:
        raise ValueError(f"Failed to get file metadata: {str(e)}")

def delete_file_from_gridfs(file_id: str) -> bool:
    """
    Delete file from GridFS
    
    Args:
        file_id: GridFS file ID
        
    Returns:
        bool: True if deleted successfully
    """
    try:
        fs.delete(ObjectId(file_id))
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {e}")
        return False

# ------------------------------
# Image Processing
# ------------------------------

def validate_image_file(file_path: str) -> bool:
    """
    Validate if file is a valid image
    
    Args:
        file_path: Path to image file
        
    Returns:
        bool: True if valid image
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_info(file_path: str) -> dict:
    """
    Get image information
    
    Args:
        file_path: Path to image file
        
    Returns:
        dict: Image information
    """
    try:
        with Image.open(file_path) as img:
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height
            }
    except Exception as e:
        raise ValueError(f"Failed to get image info: {str(e)}")

# ------------------------------
# Cleanup and Utilities
# ------------------------------

def cleanup_temporary_files():
    """Manually trigger cleanup of temporary files"""
    _cleanup_temp_files()

def get_temp_file_count() -> int:
    """Get number of tracked temporary files"""
    return len(_temp_files)

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