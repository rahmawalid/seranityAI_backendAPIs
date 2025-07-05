"""
Doctor Notes Repository Layer - Data Access Layer
Handles all database operations for doctor notes functionality
Fixed to align with patient, doctor, and report repository patterns
"""

from bson import ObjectId
from mongoengine import DoesNotExist
from model.patient_model import Patient, Session
from config import fs 
from typing import List, Optional, Tuple, Dict, Any, Union
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoctorNotesRepository:
    """Repository class for doctor notes data access operations"""
    
    def __init__(self):
        self.fs = fs  # GridFS instance for file storage
    
    # ================================
    # HELPER FUNCTIONS (Aligned with other repos)
    # ================================
    
    def _handle_gridfs_operation(self, operation_func, error_message: str):
        """Helper to handle GridFS operations with proper error handling"""
        try:
            return operation_func()
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            raise ValueError(f"{error_message}: {str(e)}")
    
    def _normalize_patient_id(self, patient_id: Union[str, int]) -> int:
        """Convert patient ID to integer, handling both 'P123' and '123' formats"""
        try:
            if isinstance(patient_id, str):
                return int(patient_id.strip("P"))
            return int(patient_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid patient ID format: {patient_id}")
    
    def _validate_file_id(self, file_id: str) -> ObjectId:
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
    
    def _get_patient_by_id(self, patient_id: Union[str, int]) -> Patient:
        """Helper to get patient with proper error handling"""
        try:
            normalized_id = self._normalize_patient_id(patient_id)
            patient = Patient.objects(patientID=normalized_id).first()
            if not patient:
                raise DoesNotExist(f"Patient with ID {patient_id} not found")
            return patient
        except (DoesNotExist, ValueError) as e:
            raise e
    
    def _find_session_index(self, patient: Patient, session_id: int) -> int:
        """Find session index by session_id"""
        for i, session in enumerate(patient.sessions):
            if session.session_id == session_id:
                return i
        raise ValueError(f"Session {session_id} not found")
    
    def _validate_session_id(self, session_id: Union[str, int]) -> int:
        """Validate and convert session ID to integer"""
        try:
            return int(session_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid session ID format: {session_id}")
    
    def get_notebook_integration_data(self, patient_id: Union[str, int], session_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get data formatted for notebook integration
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            
        Returns:
            Dict containing patient info, session info, and doctor notes files
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session, _ = self.find_session_by_id(patient, validated_session_id)
            
            # Get doctor notes files
            doctor_notes_files = self.get_doctor_notes_from_session(patient_id, session_id)
            
            # Format patient info for notebook
            patient_info = {
                'Patient ID': str(patient.patientID),
                'Full Name': patient.personalInfo.full_name,
                'Age': self._calculate_age(patient.personalInfo.date_of_birth),
                'Gender': patient.personalInfo.gender,
                'Occupation': patient.personalInfo.occupation,
                'Marital Status': patient.personalInfo.marital_status,
                'Reason for Therapy': patient.personalInfo.therapy_info.reason_for_therapy,
                'Physical Health Conditions': patient.personalInfo.health_info.physical_health_conditions,
                'Family History of Mental Illness': patient.personalInfo.health_info.family_history_of_mental_illness,
                'Substance Use': patient.personalInfo.health_info.substance_use,
                'Current Medications': patient.personalInfo.health_info.current_medications,
            }
            
            # Format session info for notebook
            session_info = {
                'Session Number': session.session_id,
                'Date of Session': session.date.strftime('%Y-%m-%d') if session.date else None,
                'Duration': session.duration,
                'Session Type': session.session_type,
                'Patient ID': str(patient.patientID),
            }
            
            return {
                'patient_info': patient_info,
                'session_info': session_info,
                'doctor_notes_files': doctor_notes_files,
                'has_doctor_notes': len(doctor_notes_files) > 0,
                'notes_count': len(doctor_notes_files)
            }
            
        except Exception as e:
            raise ValueError(f"Failed to get notebook integration data: {str(e)}")
    
    def _calculate_age(self, date_of_birth: str) -> int:
        """Calculate age from date of birth string"""
        try:
            if not date_of_birth:
                return 0
            
            from datetime import datetime
            
            # Try multiple date formats
            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']
            birth_date = None
            
            for fmt in date_formats:
                try:
                    birth_date = datetime.strptime(date_of_birth, fmt)
                    break
                except ValueError:
                    continue
            
            if not birth_date:
                return 0
                
            today = datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except Exception:
            return 0

    # ================================
    # FILE STORAGE OPERATIONS
    # ================================
    
    def save_file_to_gridfs(self, file_storage) -> str:
        """
        Save doctor notes file to GridFS
        
        Args:
            file_storage: File storage object with read() method
            
        Returns:
            str: GridFS file ID
            
        Raises:
            ValueError: If file save fails
        """
        try:
            if not file_storage:
                raise ValueError("File storage object is required")
            
            content_type = getattr(file_storage, 'content_type', 'image/jpeg') or 'image/jpeg'
            filename = getattr(file_storage, 'filename', 'doctor_note.jpg')
            
            def save_operation():
                return self.fs.put(
                    file_storage, 
                    filename=filename, 
                    content_type=content_type
                )
            
            file_id = self._handle_gridfs_operation(
                save_operation,
                "Failed to save doctor notes file to GridFS"
            )
            
            logger.info(f"Successfully saved doctor notes file {filename} with ID {file_id}")
            return str(file_id)
            
        except Exception as e:
            raise ValueError(f"Failed to save file to GridFS: {str(e)}")
    
    def get_file_from_gridfs(self, file_id: str):
        """
        Retrieve file from GridFS by ID
        
        Args:
            file_id: GridFS file ID
            
        Returns:
            GridFS file object
            
        Raises:
            ValueError: If file not found or retrieval fails
        """
        try:
            object_id = self._validate_file_id(file_id)
            
            def get_operation():
                return self.fs.get(object_id)
            
            return self._handle_gridfs_operation(
                get_operation,
                f"Failed to retrieve file {file_id} from GridFS"
            )
            
        except Exception as e:
            raise ValueError(f"Failed to get file from GridFS: {str(e)}")
    
    def delete_file_from_gridfs(self, file_id: str) -> bool:
        """
        Delete file from GridFS
        
        Args:
            file_id: GridFS file ID
            
        Returns:
            bool: True if deleted successfully
            
        Raises:
            ValueError: If deletion fails
        """
        try:
            object_id = self._validate_file_id(file_id)
            
            def delete_operation():
                self.fs.delete(object_id)
                return True
            
            result = self._handle_gridfs_operation(
                delete_operation,
                f"Failed to delete file {file_id} from GridFS"
            )
            
            logger.info(f"Successfully deleted file {file_id} from GridFS")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise ValueError(f"Failed to delete file: {str(e)}")
    
    def get_file_data_for_download(self, file_id: str) -> Tuple[str, bytes, str]:
        """
        Get file data for download
        
        Args:
            file_id: GridFS file ID
            
        Returns:
            Tuple of (filename, file_data, content_type)
            
        Raises:
            ValueError: If file retrieval fails
        """
        try:
            file_obj = self.get_file_from_gridfs(file_id)
            
            filename = file_obj.filename or f"doctor_note_{file_id}"
            content = file_obj.read()
            content_type = getattr(file_obj, 'content_type', 'image/jpeg') or 'image/jpeg'
            
            return filename, content, content_type
            
        except Exception as e:
            raise ValueError(f"Failed to get file data for download: {str(e)}")
    
    def check_file_exists_in_gridfs(self, file_id: str) -> bool:
        """
        Check if file exists in GridFS
        
        Args:
            file_id: GridFS file ID
            
        Returns:
            bool: True if file exists
        """
        try:
            self.get_file_from_gridfs(file_id)
            return True
        except Exception:
            return False
    
    # ================================
    # PATIENT SESSION OPERATIONS
    # ================================
    
    def get_patient_by_id(self, patient_id: Union[str, int]) -> Patient:
        """
        Get patient by ID with proper error handling
        
        Args:
            patient_id: Patient ID (can be 'P123' or '123' format)
            
        Returns:
            Patient object
            
        Raises:
            ValueError: If patient not found
        """
        return self._get_patient_by_id(patient_id)
    
    def find_session_by_id(self, patient: Patient, session_id: Union[str, int]) -> Tuple[Session, int]:
        """
        Find session by ID and return session object and index
        
        Args:
            patient: Patient object
            session_id: Session ID
            
        Returns:
            Tuple of (session_object, session_index)
            
        Raises:
            ValueError: If session not found
        """
        try:
            validated_session_id = self._validate_session_id(session_id)
            session_index = self._find_session_index(patient, validated_session_id)
            return patient.sessions[session_index], session_index
        except Exception as e:
            raise ValueError(f"Session not found: {str(e)}")
    
    # ================================
    # DOCTOR NOTES CRUD OPERATIONS
    # ================================
    
    def attach_doctor_notes_to_session(self, patient_id: Union[str, int], session_id: Union[str, int], file_ids: List[str]) -> bool:
        """
        Attach doctor notes to a patient session
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            file_ids: List of GridFS file IDs
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If patient/session not found or operation fails
        """
        try:
            if not file_ids:
                raise ValueError("File IDs list cannot be empty")
            
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session_index = self._find_session_index(patient, validated_session_id)
            
            # Validate all file IDs exist
            validated_file_ids = []
            for file_id in file_ids:
                object_id = self._validate_file_id(file_id)
                # Check if file exists in GridFS
                if not self.check_file_exists_in_gridfs(file_id):
                    raise ValueError(f"File {file_id} does not exist in GridFS")
                validated_file_ids.append(object_id)
            
            # Update the session with doctor notes
            if not patient.sessions[session_index].doctor_notes_images:
                patient.sessions[session_index].doctor_notes_images = []
            
            # Extend existing list with new files
            patient.sessions[session_index].doctor_notes_images.extend(validated_file_ids)
            
            # Save the patient
            patient.save()
            
            logger.info(f"Successfully attached {len(file_ids)} doctor notes to patient {patient_id}, session {session_id}")
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to attach doctor notes: {str(e)}")
    
    def get_doctor_notes_from_session(self, patient_id: Union[str, int], session_id: Union[str, int]) -> List[Any]:
        """
        Get all doctor notes from a specific session
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            
        Returns:
            List of GridFS file objects
            
        Raises:
            ValueError: If patient/session not found
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session, _ = self.find_session_by_id(patient, validated_session_id)
            
            doctor_notes_files = []
            doctor_notes_ids = session.doctor_notes_images or []
            
            for note_id in doctor_notes_ids:
                try:
                    file_obj = self.get_file_from_gridfs(str(note_id))
                    doctor_notes_files.append(file_obj)
                except Exception as e:
                    logger.warning(f"Could not retrieve doctor note file {note_id}: {e}")
                    continue
            
            return doctor_notes_files
            
        except Exception as e:
            raise ValueError(f"Failed to get doctor notes from session: {str(e)}")
    
    def get_doctor_notes_ids_from_session(self, patient_id: Union[str, int], session_id: Union[str, int]) -> List[str]:
        """
        Get doctor notes file IDs from a specific session
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            
        Returns:
            List of file ID strings
            
        Raises:
            ValueError: If patient/session not found
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session, _ = self.find_session_by_id(patient, validated_session_id)
            
            doctor_notes_ids = session.doctor_notes_images or []
            return [str(note_id) for note_id in doctor_notes_ids]
            
        except Exception as e:
            raise ValueError(f"Failed to get doctor notes IDs: {str(e)}")
    
    def remove_doctor_note_from_session(self, patient_id: Union[str, int], session_id: Union[str, int], file_id: str) -> bool:
        """
        Remove a specific doctor note from a session
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            file_id: GridFS file ID to remove
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If patient/session not found or operation fails
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session_index = self._find_session_index(patient, validated_session_id)
            validated_file_id = self._validate_file_id(file_id)
            
            doctor_notes_ids = patient.sessions[session_index].doctor_notes_images or []
            
            # Check if file ID exists in the session
            if validated_file_id not in doctor_notes_ids:
                raise ValueError(f"File {file_id} not found in session {session_id}")
            
            # Remove the specific file ID
            updated_ids = [oid for oid in doctor_notes_ids if oid != validated_file_id]
            
            # Update the session
            patient.sessions[session_index].doctor_notes_images = updated_ids
            patient.save()
            
            # Delete from GridFS
            try:
                self.delete_file_from_gridfs(file_id)
            except Exception as e:
                logger.warning(f"Failed to delete file {file_id} from GridFS: {e}")
            
            logger.info(f"Successfully removed doctor note {file_id} from patient {patient_id}, session {session_id}")
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to remove doctor note: {str(e)}")
    
    def clear_all_doctor_notes_from_session(self, patient_id: Union[str, int], session_id: Union[str, int]) -> bool:
        """
        Clear all doctor notes from a session
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If patient/session not found
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session_index = self._find_session_index(patient, validated_session_id)
            
            # Get existing file IDs for cleanup
            doctor_notes_ids = patient.sessions[session_index].doctor_notes_images or []
            
            # Clear the list
            patient.sessions[session_index].doctor_notes_images = []
            patient.save()
            
            # Delete files from GridFS
            deleted_count = 0
            for note_id in doctor_notes_ids:
                try:
                    self.delete_file_from_gridfs(str(note_id))
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete file {note_id} from GridFS: {e}")
            
            logger.info(f"Successfully cleared {deleted_count} doctor notes from patient {patient_id}, session {session_id}")
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to clear doctor notes: {str(e)}")
    
    # ================================
    # QUERY OPERATIONS
    # ================================
    
    def session_has_doctor_notes(self, patient_id: Union[str, int], session_id: Union[str, int]) -> bool:
        """
        Check if a session has doctor notes attached
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            
        Returns:
            bool: True if session has doctor notes
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session, _ = self.find_session_by_id(patient, validated_session_id)
            
            doctor_notes = session.doctor_notes_images or []
            return len(doctor_notes) > 0
            
        except Exception as e:
            logger.warning(f"Error checking if session has doctor notes: {e}")
            return False
    
    def get_doctor_notes_count(self, patient_id: Union[str, int], session_id: Union[str, int]) -> int:
        """
        Get count of doctor notes in a session
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            
        Returns:
            int: Number of doctor notes
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session, _ = self.find_session_by_id(patient, validated_session_id)
            
            doctor_notes = session.doctor_notes_images or []
            return len(doctor_notes)
            
        except Exception as e:
            logger.warning(f"Error getting doctor notes count: {e}")
            return 0
    
    def get_patient_doctor_notes_summary(self, patient_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get summary of all doctor notes for a patient across all sessions
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Dict containing summary information
            
        Raises:
            ValueError: If patient not found
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            
            total_notes = 0
            sessions_with_notes = 0
            notes_by_session = {}
            
            for session in patient.sessions:
                doctor_notes = session.doctor_notes_images or []
                if doctor_notes:
                    sessions_with_notes += 1
                    total_notes += len(doctor_notes)
                    notes_by_session[session.session_id] = {
                        "count": len(doctor_notes),
                        "file_ids": [str(note_id) for note_id in doctor_notes],
                        "session_date": session.date.isoformat() if session.date else None,
                        "session_type": session.session_type
                    }
            
            return {
                "patient_id": str(patient.patientID),
                "total_notes": total_notes,
                "sessions_with_notes": sessions_with_notes,
                "total_sessions": len(patient.sessions),
                "notes_by_session": notes_by_session
            }
            
        except Exception as e:
            raise ValueError(f"Failed to get patient doctor notes summary: {str(e)}")
    
    # ================================
    # SESSION METADATA OPERATIONS
    # ================================
    
    def update_session_analysis_type(self, patient_id: Union[str, int], session_id: Union[str, int], analysis_type: str = "comprehensive") -> bool:
        """
        Update session to indicate analysis type
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            analysis_type: Type of analysis performed
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If patient/session not found
        """
        try:
            valid_analysis_types = [
                "comprehensive_with_notes", "speech_with_notes", 
                "comprehensive", "speech_only", "basic"
            ]
            
            if analysis_type not in valid_analysis_types:
                raise ValueError(f"Invalid analysis type: {analysis_type}. Valid types: {valid_analysis_types}")
            
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session_index = self._find_session_index(patient, validated_session_id)
            
            # Initialize feature_data if it doesn't exist
            if not patient.sessions[session_index].feature_data:
                patient.sessions[session_index].feature_data = {}
            
            # Update analysis metadata
            patient.sessions[session_index].feature_data["analysis_type"] = analysis_type
            patient.sessions[session_index].feature_data["has_doctor_notes"] = self.session_has_doctor_notes(patient_id, session_id)
            
            patient.save()
            
            logger.info(f"Updated session {session_id} analysis type to {analysis_type} for patient {patient_id}")
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to update session analysis type: {str(e)}")
    
    def get_session_analysis_capabilities(self, patient_id: Union[str, int], session_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get analysis capabilities for a session based on available data
        
        Args:
            patient_id: Patient ID
            session_id: Session ID
            
        Returns:
            Dict containing analysis capabilities
            
        Raises:
            ValueError: If patient/session not found
        """
        try:
            patient = self._get_patient_by_id(patient_id)
            validated_session_id = self._validate_session_id(session_id)
            session, _ = self.find_session_by_id(patient, validated_session_id)
            
            # Get existing analysis data
            fer_data = session.feature_data.get("FER", {}) if session.feature_data else {}
            speech_data = session.feature_data.get("Speech", {}) if session.feature_data else {}
            doctor_notes = session.doctor_notes_images or []
            
            capabilities = {
                "patient_id": str(patient.patientID),
                "session_id": session.session_id,
                "has_fer": bool(fer_data.get("fer_excel")),
                "has_speech": bool(speech_data.get("speech_excel")),
                "has_doctor_notes": len(doctor_notes) > 0,
                "fer_files": {
                    "excel": str(fer_data.get("fer_excel")) if fer_data.get("fer_excel") else None,
                    "images": [str(img_id) for img_id in fer_data.get("plot_images", [])]
                },
                "speech_files": {
                    "excel": str(speech_data.get("speech_excel")) if speech_data.get("speech_excel") else None
                },
                "doctor_notes_files": [str(note_id) for note_id in doctor_notes],
                "doctor_notes_count": len(doctor_notes),
                "analysis_type": "unknown"
            }
            
            # Determine analysis type based on available data
            if capabilities["has_fer"] and capabilities["has_speech"] and capabilities["has_doctor_notes"]:
                capabilities["analysis_type"] = "comprehensive_with_notes"
            elif capabilities["has_speech"] and capabilities["has_doctor_notes"]:
                capabilities["analysis_type"] = "speech_with_notes"
            elif capabilities["has_fer"] and capabilities["has_speech"]:
                capabilities["analysis_type"] = "comprehensive"
            elif capabilities["has_speech"]:
                capabilities["analysis_type"] = "speech_only"
            elif capabilities["has_doctor_notes"]:
                capabilities["analysis_type"] = "notes_only"
            else:
                capabilities["analysis_type"] = "basic"
            
            # Add recommendation
            if not capabilities["has_doctor_notes"]:
                capabilities["recommendation"] = "Upload doctor notes for enhanced analysis"
            elif not capabilities["has_fer"] and not capabilities["has_speech"]:
                capabilities["recommendation"] = "Add FER or speech analysis for comprehensive insights"
            else:
                capabilities["recommendation"] = "All data types available for comprehensive analysis"
            
            return capabilities
            
        except Exception as e:
            raise ValueError(f"Failed to get session analysis capabilities: {str(e)}")
    
    # ================================
    # VALIDATION AND UTILITY METHODS
    # ================================
    
    def validate_doctor_notes_upload(self, file_ids: List[str]) -> Dict[str, Any]:
        """
        Validate doctor notes files before upload
        
        Args:
            file_ids: List of file IDs to validate
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "validated_files": [],
            "total_files": len(file_ids)
        }
        
        if not file_ids:
            validation_result["valid"] = False
            validation_result["errors"].append("No files provided for validation")
            return validation_result
        
        for file_id in file_ids:
            try:
                # Validate file ID format
                self._validate_file_id(file_id)
                
                # Check if file exists in GridFS
                if self.check_file_exists_in_gridfs(file_id):
                    validation_result["validated_files"].append(file_id)
                else:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"File {file_id} does not exist in GridFS")
                    
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid file ID {file_id}: {str(e)}")
        
        if len(validation_result["validated_files"]) != len(file_ids):
            validation_result["warnings"].append(f"Only {len(validation_result['validated_files'])} out of {len(file_ids)} files are valid")
        
        return validation_result