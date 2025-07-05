import datetime
import logging
import gridfs
from datetime import timedelta
from bson import ObjectId
from pymongo import MongoClient
from mongoengine import DoesNotExist, ValidationError, NotUniqueError
from model.doctor_model import Doctor, ScheduledSession
from typing import List
import uuid
from utils.email_utils import send_verification_email

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

# Constants
TOKEN_EXPIRY_HOURS = 1

# ------------------------------
# Helper Functions
# ------------------------------

def _handle_gridfs_operation(operation_func, error_message: str):
    """Helper to handle GridFS operations with proper error handling"""
    try:
        return operation_func()
    except Exception as e:
        raise ValueError(f"{error_message}: {str(e)}")

def _is_token_expired(token_created_at: datetime.datetime) -> bool:
    """Check if verification token is expired (1 hour expiry)"""
    if not token_created_at:
        return True
    
    expiry_time = token_created_at + timedelta(hours=TOKEN_EXPIRY_HOURS)
    return datetime.datetime.utcnow() > expiry_time

def _check_doctor_email_uniqueness(email: str):
    """Check if email already exists in the system"""
    existing_doctor = Doctor.objects(personal_info__email=email).first()
    if existing_doctor:
        raise ValueError("Email already exists, please login or use a different email.")

def _check_doctor_phone_uniqueness(phone: str):
    """Check if phone number already exists in the system"""
    existing_doctor = Doctor.objects(personal_info__phone_number=phone).first()
    if existing_doctor:
        raise ValueError("Phone number already exists, please use a different number.")

# ------------------------------
# File Upload to GridFS
# ------------------------------
def save_file_to_gridfs(file_storage, content_type: str) -> str:
    """
    Save a file to GridFS and return its ObjectId as a string.
    """
    def save_operation():
        return str(
            fs.put(file_storage, filename=file_storage.filename, content_type=content_type)
        )
    
    return _handle_gridfs_operation(
        save_operation, 
        "Failed to save file to GridFS"
    )

# ------------------------------
# Create Doctor
# ------------------------------
def create_doctor(doctor_data: dict) -> Doctor:
    """Create a new doctor with email verification"""
    try:
        if not doctor_data or "password" not in doctor_data:
            raise ValueError("Password is required.")

        # Extract email and phone from nested structure
        personal_info = doctor_data.get("personal_info", {})
        email = personal_info.get("email")
        phone = personal_info.get("phone_number")
        
        if not email:
            raise ValueError("Email is required.")
        if not phone:
            raise ValueError("Phone number is required.")

        # Check for uniqueness using correct field paths
        _check_doctor_email_uniqueness(email)
        _check_doctor_phone_uniqueness(phone)

        # Extract and hash password
        raw_password = doctor_data.pop("password")
        doctor = Doctor(**doctor_data)
        doctor.set_password(raw_password)

        # Generate verification token with expiration
        token = str(uuid.uuid4())
        doctor.verification_token = token
        doctor.token_created_at = datetime.datetime.utcnow()
        doctor.email_verified = False
        doctor.save()

        # Send verification email
        send_verification_email(email, token)

        return doctor

    except (ValueError, ValidationError, NotUniqueError) as e:
        raise e
    except Exception as e:
        raise ValueError(f"Failed to create doctor: {str(e)}")

# ------------------------------
# Email Verification
# ------------------------------
def verify_doctor_email(token: str) -> Doctor:
    """
    Verify doctor email using verification token
    
    Args:
        token: Email verification token
        
    Returns:
        Doctor: Verified doctor object
        
    Raises:
        ValueError: If token is invalid, expired, or doctor not found
    """
    try:
        # Find doctor by token
        doctor = Doctor.objects(verification_token=token).first()
        if not doctor:
            raise ValueError("Invalid verification token.")
        
        # Check if token is expired
        if _is_token_expired(doctor.token_created_at):
            raise ValueError("Verification token has expired. Please request a new verification email.")
        
        # Check if already verified
        if doctor.email_verified:
            raise ValueError("Email is already verified.")
        
        # Verify the email
        doctor.email_verified = True
        doctor.verification_token = None  # Clear the token
        doctor.token_created_at = None    # Clear the timestamp
        doctor.save()
        
        return doctor
        
    except (ValidationError, DoesNotExist) as e:
        raise ValueError(f"Email verification failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Email verification failed: {str(e)}")

def resend_verification_email(email: str) -> bool:
    """
    Resend verification email to doctor
    
    Args:
        email: Doctor's email address
        
    Returns:
        bool: True if email was sent successfully
        
    Raises:
        ValueError: If doctor not found or already verified
    """
    try:
        # Find doctor by email using correct field path
        doctor = Doctor.objects(personal_info__email=email).first()
        if not doctor:
            raise ValueError("Doctor not found with this email address.")
        
        # Check if already verified
        if doctor.email_verified:
            raise ValueError("Email is already verified.")
        
        # Generate new token
        token = str(uuid.uuid4())
        doctor.verification_token = token
        doctor.token_created_at = datetime.datetime.utcnow()
        doctor.save()
        
        # Send verification email
        send_verification_email(email, token)
        
        return True
        
    except Exception as e:
        raise ValueError(f"Failed to resend verification email: {str(e)}")

# ------------------------------
# Get Doctor by ID
# ------------------------------
# def get_doctor_by_id(doctor_ID: str) -> Doctor:
#     """
#     Get doctor by doctor_ID with proper error handling.
#     """
#     try:
#         logger.info(f"🔍 Looking for doctor with ID: '{doctor_ID}'")
        
#         # Direct query first
#         doctor = Doctor.objects(doctor_ID=doctor_ID).first()
#         if doctor:
#             logger.info(f"✅ Found doctor: {doctor.personal_info.email if doctor.personal_info else 'No email'}")
#             return doctor
        
#         # If not found, try case-insensitive search
#         doctor = Doctor.objects(doctor_ID__iexact=doctor_ID).first()
#         if doctor:
#             logger.info(f"✅ Found doctor (case-insensitive): {doctor.personal_info.email if doctor.personal_info else 'No email'}")
#             return doctor
        
#         # If still not found, log available doctors for debugging
#         logger.error(f"❌ Doctor {doctor_ID} not found")
        
#         # List available doctors (safely)
#         try:
#             all_doctors = Doctor.objects.only('doctor_ID', 'personal_info__email')
#             available_doctors = []
#             for d in all_doctors:
#                 try:
#                     doc_id = getattr(d, 'doctor_ID', 'UNKNOWN')
#                     email = 'no_email'
#                     if hasattr(d, 'personal_info') and d.personal_info:
#                         email = getattr(d.personal_info, 'email', 'no_email')
#                     available_doctors.append(f"{doc_id} ({email})")
#                 except Exception:
#                     available_doctors.append("Doctor with corrupted data")
            
#             logger.error(f"Available doctors: {available_doctors}")
#         except Exception as list_error:
#             logger.error(f"Could not list available doctors: {list_error}")
        
#         raise DoesNotExist(f"Doctor with ID '{doctor_ID}' not found")
        
#     except DoesNotExist:
#         raise ValueError(f"Doctor with ID '{doctor_ID}' not found")
#     except ValidationError as e:
#         logger.error(f"❌ Validation error: {e}")
#         raise ValueError(f"Invalid doctor ID format: {doctor_ID}")
#     except Exception as e:
#         logger.error(f"❌ Unexpected error: {e}")
#         raise ValueError(f"Database error: {str(e)}")
    
def get_doctor_by_id(doctor_ID: str) -> Doctor:
    """
    Get doctor by doctor_ID using raw query to bypass MongoEngine bug.
    """
    try:
        logger.info(f"🔍 Looking for doctor with ID: '{doctor_ID}'")
        
        # Use __raw__ query to bypass the MongoEngine compilation issue
        doctor = Doctor.objects(__raw__={"doctor_ID": doctor_ID}).first()
        if doctor:
            logger.info(f"✅ Found doctor using raw query: {doctor.personal_info.email if doctor.personal_info else 'No email'}")
            return doctor
        
        # Fallback: Direct ObjectId lookup if we can find the document
        try:
            from pymongo import MongoClient
            from bson import ObjectId
            
            client = MongoClient("mongodb://localhost:27017")
            db = client["seranityAI"]
            doctors_collection = db["Doctors"]
            
            raw_doc = doctors_collection.find_one({"doctor_ID": doctor_ID})
            if raw_doc:
                # Get the document through MongoEngine using ObjectId
                doc_id = raw_doc['_id']
                doctor = Doctor.objects(id=doc_id).first()
                if doctor:
                    logger.info(f"✅ Found doctor using direct MongoDB + ObjectId lookup")
                    return doctor
        except Exception as fallback_error:
            logger.warning(f"Fallback method failed: {fallback_error}")
        
        # If not found, log available doctors for debugging
        logger.error(f"❌ Doctor {doctor_ID} not found")
        
        try:
            # Use raw query to list available doctors
            all_doctors = Doctor.objects(__raw__={}).only('doctor_ID', 'personal_info')
            available_doctors = []
            for d in all_doctors:
                try:
                    doc_id = getattr(d, 'doctor_ID', 'UNKNOWN')
                    email = 'no_email'
                    if hasattr(d, 'personal_info') and d.personal_info:
                        email = getattr(d.personal_info, 'email', 'no_email')
                    available_doctors.append(f"{doc_id} ({email})")
                except Exception:
                    available_doctors.append("Doctor with corrupted data")
            
            logger.error(f"Available doctors: {available_doctors}")
        except Exception as list_error:
            logger.error(f"Could not list available doctors: {list_error}")
        
        raise DoesNotExist(f"Doctor with ID '{doctor_ID}' not found")
        
    except DoesNotExist:
        raise ValueError(f"Doctor with ID '{doctor_ID}' not found")
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        raise ValueError(f"Invalid doctor ID format: {doctor_ID}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise ValueError(f"Database error: {str(e)}")

# ------------------------------
# Update Doctor
# ------------------------------
def update_doctor(doctor_ID: str, update_data: dict) -> Doctor:
    """
    Updates doctor fields by key. Handles nested fields and list updates.
    """
    try:
        doctor = get_doctor_by_id(doctor_ID)
        
        # Handle password separately
        if "password" in update_data:
            raw_password = update_data.pop("password")
            doctor.set_password(raw_password)
        
        # Handle email uniqueness check if email is being updated
        if "personal_info" in update_data and "email" in update_data["personal_info"]:
            new_email = update_data["personal_info"]["email"]
            current_email = doctor.personal_info.email
            
            # Only check uniqueness if email is actually changing
            if new_email != current_email:
                _check_doctor_email_uniqueness(new_email)
                # Reset email verification if email changes
                doctor.email_verified = False
                token = str(uuid.uuid4())
                doctor.verification_token = token
                doctor.token_created_at = datetime.datetime.utcnow()
                send_verification_email(new_email, token)
        
        # Handle phone uniqueness check if phone is being updated
        if "personal_info" in update_data and "phone_number" in update_data["personal_info"]:
            new_phone = update_data["personal_info"]["phone_number"]
            current_phone = doctor.personal_info.phone_number
            
            # Only check uniqueness if phone is actually changing
            if new_phone != current_phone:
                _check_doctor_phone_uniqueness(new_phone)
        
        # Update other fields
        for key, value in update_data.items():
            if hasattr(doctor, key):
                setattr(doctor, key, value)
        
        doctor.save()
        return doctor
        
    except (DoesNotExist, ValidationError) as e:
        raise ValueError(f"Failed to update doctor: {str(e)}")
    except Exception as e:
        raise ValueError(f"Update failed: {str(e)}")

# ------------------------------
# Delete Doctor
# ------------------------------
def delete_doctor(doctor_ID: str) -> None:
    """Delete a doctor by ID"""
    try:
        result = Doctor.objects(doctor_ID=doctor_ID).delete()
        if result == 0:
            raise ValueError(f"Doctor with ID '{doctor_ID}' not found")
    except Exception as e:
        raise ValueError(f"Failed to delete doctor: {str(e)}")

# ------------------------------
# Login Doctor
# ------------------------------
def login_doctor(email: str, password: str) -> Doctor:
    """
    Authenticate doctor login with email verification check
    """
    try:
        # Find doctor by email using correct field path
        doctor = Doctor.objects(personal_info__email=email).first()
        if not doctor:
            raise ValueError("Doctor not found.")

        if not doctor.email_verified:
            raise ValueError("Email not verified. Please check your inbox.")

        if doctor.check_password(password):
            return doctor
        else:
            raise ValueError("Invalid credentials.")

    except DoesNotExist:
        raise ValueError("Doctor not found.")
    except Exception as e:
        raise ValueError(f"Login failed: {str(e)}")

# ------------------------------
# Update Doctor Password
# ------------------------------
def update_doctor_password(doctor_ID: str, old_password: str, new_password: str) -> bool:
    """Update doctor password after verifying old password"""
    try:
        doctor = get_doctor_by_id(doctor_ID)
        if not doctor.check_password(old_password):
            raise ValueError("Old password is incorrect.")
        
        doctor.set_password(new_password)
        doctor.save()
        return True
        
    except Exception as e:
        raise ValueError(f"Password update failed: {str(e)}")

# ------------------------------
# Schedule Session
# ------------------------------
def schedule_session_for_doctor(
    doctor_id: str, patient_id: int, when_iso: str, notes: str
) -> ScheduledSession:
    """Schedule a new session for a doctor with enhanced error handling"""
    try:
        logger.info(f"🔍 SCHEDULING: doctor_id='{doctor_id}', patient_id={patient_id}")
        
        # Validate inputs
        if not doctor_id or not isinstance(doctor_id, str):
            raise ValueError("Invalid doctor_id provided")
        
        if not isinstance(patient_id, int) or patient_id <= 0:
            raise ValueError("Invalid patient_id provided")
        
        if not when_iso or not isinstance(when_iso, str):
            raise ValueError("Invalid datetime provided")
        
        # Get doctor with enhanced error handling
        try:
            doctor = get_doctor_by_id(doctor_id)
            logger.info(f"✅ Doctor found: {doctor.personal_info.email if doctor.personal_info else 'No email'}")
        except Exception as e:
            logger.error(f"❌ Failed to find doctor '{doctor_id}': {e}")
            raise ValueError(f"Doctor '{doctor_id}' not found. Please check the doctor ID.")

        # Parse datetime
        try:
            when = datetime.datetime.fromisoformat(when_iso.strip())
        except ValueError as e:
            logger.error(f"❌ Invalid datetime format: {when_iso}")
            raise ValueError(f"Invalid datetime format: {when_iso}. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

        # Validate datetime is in the future
        if when <= datetime.datetime.now():
            raise ValueError("Cannot schedule sessions in the past.")

        # Create session
        session = ScheduledSession(
            patientID=patient_id,
            datetime=when,
            notes=notes or "",
        )
        
        # Initialize scheduledSessions if it doesn't exist
        if not hasattr(doctor, 'scheduledSessions') or doctor.scheduledSessions is None:
            doctor.scheduledSessions = []
        
        # Add to doctor's sessions
        doctor.scheduledSessions.append(session)
        
        # Save with error handling
        try:
            doctor.save()
            logger.info(f"✅ Session scheduled successfully for doctor {doctor_id}, patient {patient_id}")
        except Exception as save_error:
            logger.error(f"❌ Failed to save session: {save_error}")
            raise ValueError(f"Failed to save session to database: {str(save_error)}")
        
        return session

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected scheduling error: {e}")
        raise ValueError(f"Failed to schedule session: {str(e)}")

# ------------------------------
# Get Scheduled Sessions
# ------------------------------
def get_scheduled_sessions_for_doctor(doctor_id: str) -> List[ScheduledSession]:
    """Get all scheduled sessions for a doctor"""
    try:
        doctor = get_doctor_by_id(doctor_id)
        return doctor.scheduledSessions or []
    except Exception as e:
        raise ValueError(f"Failed to get scheduled sessions: {str(e)}")

# ------------------------------
# Token Validation Utilities
# ------------------------------
def check_token_validity(token: str) -> dict:
    """
    Check token validity without verifying email
    
    Args:
        token: Verification token to check
        
    Returns:
        dict: Token status information
    """
    try:
        doctor = Doctor.objects(verification_token=token).first()
        if not doctor:
            return {
                "valid": False,
                "reason": "Token not found",
                "expired": False
            }
        
        is_expired = _is_token_expired(doctor.token_created_at)
        
        return {
            "valid": not is_expired,
            "reason": "Token expired" if is_expired else "Token valid",
            "expired": is_expired,
            "doctor_email": doctor.personal_info.email if doctor.personal_info else None
        }
        
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Error checking token: {str(e)}",
            "expired": False
        }