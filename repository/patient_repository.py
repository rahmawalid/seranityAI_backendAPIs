import gridfs
from bson import ObjectId
from pymongo import MongoClient
from mongoengine import DoesNotExist, ValidationError, NotUniqueError
import pandas as pd
import datetime
from model.patient_model import Patient, Session
from model.doctor_model import Doctor
import io



# -------------------------------
# MongoDB Setup (GridFS)
# -------------------------------
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

# -------------------------------
# Helper Functions
# -------------------------------

def _normalize_patient_id(patient_id):
    """Convert patient ID to integer, handling both 'P123' and '123' formats"""
    try:
        if isinstance(patient_id, str):
            return int(patient_id.strip("P"))
        return int(patient_id)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid patient ID format: {patient_id}")

def _find_session_index(patient, session_id):
    """Find session index by session_id"""
    for i, session in enumerate(patient.sessions):
        if session.session_id == session_id:
            return i
    raise ValueError(f"Session {session_id} not found")

def _get_patient_by_id(patient_id):
    """Helper to get patient with proper error handling"""
    try:
        normalized_id = _normalize_patient_id(patient_id)
        patient = Patient.objects(patientID=normalized_id).first()
        if not patient:
            raise DoesNotExist(f"Patient with ID {patient_id} not found")
        return patient
    except (DoesNotExist, ValidationError, ValueError) as e:
        raise e

# -------------------------------
# Patient CRUD Operations
# -------------------------------

def create_patient_for_doctor(patient_data: dict):
    """
    Create a new Patient using the provided data dict.
    Returns the saved Patient document.
    """
    try:
        patient = Patient(**patient_data)
        patient.save()
        return patient
    except (ValidationError, NotUniqueError) as e:
        raise e

def get_patient_by_id(patient_id):
    """Get patient by ID with proper error handling"""
    return _get_patient_by_id(patient_id)

def create_patient(patient_data):
    try:
        patient = Patient(**patient_data)
        patient.save()
        return patient
    except NotUniqueError as e:
        print(e)
        raise ValueError("Patient ID must be unique!")
    except (ValidationError, Exception) as e:
        raise e

def update_patient(patient_id, patient_data):
    try:
        normalized_id = _normalize_patient_id(patient_id)
        patient = Patient.objects(patientID=normalized_id).first()
        if not patient:
            raise DoesNotExist(f"Patient with ID {patient_id} not found")
        
        # Update fields safely
        for key, value in patient_data.items():
            if hasattr(patient, key):
                setattr(patient, key, value)
        patient.save()
        return patient
    except (DoesNotExist, ValidationError, ValueError) as e:
        raise e

def delete_patient(patient_id):
    try:
        normalized_id = _normalize_patient_id(patient_id)
        result = Patient.objects(patientID=normalized_id).delete()
        if result == 0:
            raise DoesNotExist(f"Patient with ID {patient_id} not found")
    except (ValidationError, ValueError) as e:
        raise e

def get_patient_by_email(email):
    try:
        return Patient.objects.get(personalInfo__contact_information__email=email)
    except DoesNotExist as e:
        raise e

def list_patients_by_doctor(doctor_id: str):
    """
    Return a list of Patient documents belonging to the given doctor_id.
    """
    try:
        # Verify that the doctor exists
        Doctor.objects.get(doctor_ID=doctor_id)
        # Retrieve patients for this doctor
        patients = Patient.objects(doctorID=doctor_id)
        return list(patients)
    except DoesNotExist:
        raise DoesNotExist(f"Doctor with id '{doctor_id}' not found")

def get_patients_by_doctor(doctor_id: str):
    """
    Return a QuerySet of all Patient documents whose doctorID equals the given string.
    """
    try:
        return Patient.objects(doctorID=doctor_id)
    except (DoesNotExist, ValidationError):
        return Patient.objects.none()

# -------------------------------
# Audio / Video / PDF / Model‑File helpers
# -------------------------------

def save_audio_to_gridfs(file_storage):
    try:
        return str(
            fs.put(file_storage, filename=file_storage.filename, content_type="audio/mp3")
        )
    except Exception as e:
        raise ValueError(f"Failed to save audio file: {str(e)}")

def attach_audio_to_session(patient_id, session_id, file_id):
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        patient.sessions[session_index].audio_files = ObjectId(file_id)
        patient.save()
        return True
    except (DoesNotExist, ValueError) as e:
        print(f"Error attaching audio to session: {e}")
        return False

def save_video_to_gridfs(file_storage):
    try:
        return str(
            fs.put(file_storage, filename=file_storage.filename, content_type="video/mp4")
        )
    except Exception as e:
        raise ValueError(f"Failed to save video file: {str(e)}")

def attach_video_to_session(patient_id, session_id, file_id):
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        patient.sessions[session_index].video_files = ObjectId(file_id)
        patient.save()
        return True
    except (DoesNotExist, ValueError) as e:
        print(f"Error attaching video to session: {e}")
        return False

def save_pdf_to_gridfs(file_storage):
    try:
        return str(
            fs.put(
                file_storage, filename=file_storage.filename, content_type="application/pdf"
            )
        )
    except Exception as e:
        raise ValueError(f"Failed to save PDF file: {str(e)}")

def attach_pdf_to_session(patient_id, session_id, file_id):
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        patient.sessions[session_index].report = ObjectId(file_id)
        patient.save()
        return True
    except (DoesNotExist, ValueError) as e:
        print(f"Error attaching PDF to session: {e}")
        return False

def save_excel_to_gridfs(file_storage):
    try:
        return str(
            fs.put(
                file_storage,
                filename=file_storage.filename,
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        )
    except Exception as e:
        raise ValueError(f"Failed to save Excel file: {str(e)}")

def attach_model_file(patient_id, session_id, model_type, file_label, file_id):
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        patient.sessions[session_index].model_files.setdefault(model_type, {})[file_label] = ObjectId(file_id)
        patient.save()
        return True
    except (DoesNotExist, ValueError) as e:
        print(f"Error attaching model file to session: {e}")
        return False

# -------------------------------
# Session Management
# -------------------------------

def add_session_to_patient(patient_id: int, session_data: dict) -> Session:
    """
    Create and append a new Session to the given patient, auto-incrementing session_id.
    """
    try:
        patient = _get_patient_by_id(patient_id)
        
        # Determine next session_id
        if patient.sessions:
            max_id = max(sess.session_id for sess in patient.sessions)
            next_id = max_id + 1
        else:
            next_id = 1

        date_iso = session_data.get("date")
        date_obj = (
            datetime.datetime.fromisoformat(date_iso) if isinstance(date_iso, str) else None
        )

        # Construct the Session with correct field name
        new_session = Session(
            session_id=next_id,
            feature_type=session_data.get("featureType"),
            date=date_obj,
            time=session_data.get("time"),
            duration=session_data.get("duration"),
            session_type=session_data.get("sessionType"),
            text=session_data.get("text"),
            report=session_data.get("report"),
            doctor_notes_images=session_data.get("doctorNotesImages", []),  # Fixed field name
            feature_data=session_data.get("featureData"),
            audio_files=session_data.get("audioFiles"),
            video_files=session_data.get("videoFiles"),
            model_files=session_data.get("model_files"),
        )

        # Append and save
        patient.sessions.append(new_session)
        patient.save()
        return new_session
    except (DoesNotExist, ValueError) as e:
        raise e

# -------------------------------
# FER (Facial Expression Recognition) Functions
# -------------------------------

def save_fer_video_to_session(patient_id, session_id, file_path, filename):
    """Save FER video to GridFS and update session"""
    try:
        # Save to GridFS
        with open(file_path, "rb") as f:
            video_file_id = fs.put(f, filename=filename)

        # Update patient session
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        # Update feature data
        updated_feature_data = patient.sessions[session_index].feature_data or {}
        updated_feature_data["FER"] = {"video_files": ObjectId(video_file_id)}
        
        patient.sessions[session_index].feature_data = updated_feature_data
        patient.save()
        return video_file_id
    except (DoesNotExist, ValueError, IOError) as e:
        raise ValueError(f"Failed to save FER video: {str(e)}")

def update_fer_excel_reference(patient_id, session_id, excel_file_id):
    """Update FER Excel reference in session"""
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        feature_data = patient.sessions[session_index].feature_data or {}
        fer_data = feature_data.get("FER", {})
        fer_data["fer_excel"] = ObjectId(excel_file_id)
        feature_data["FER"] = fer_data
        
        patient.sessions[session_index].feature_data = feature_data
        patient.save()
        return True
    except (DoesNotExist, ValueError) as e:
        raise ValueError(f"Failed to update FER Excel reference: {str(e)}")

def update_fer_plot_reference(patient_id, session_id, plot_file_ids):
    """Save plot image ObjectIDs under feature_data.FER.plot_images"""
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        feature_data = patient.sessions[session_index].feature_data or {}
        if "FER" not in feature_data:
            feature_data["FER"] = {}

        # Add plots under FER.plot_images
        feature_data["FER"]["plot_images"] = [ObjectId(pid) for pid in plot_file_ids]
        
        patient.sessions[session_index].feature_data = feature_data
        patient.save()
    except (DoesNotExist, ValueError) as e:
        raise ValueError(f"Failed to update FER plot reference: {str(e)}")

# -------------------------------
# Speech/TOV Functions
# -------------------------------

def save_speech_video_to_session(patient_id, session_id, file_path, filename):
    """Save speech video to GridFS and update session"""
    try:
        with open(file_path, "rb") as f:
            video_file_id = fs.put(f, filename=filename)

        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        feature_data = patient.sessions[session_index].feature_data or {}
        feature_data["SpeechTOV"] = {"video_files": video_file_id}
        
        patient.sessions[session_index].feature_data = feature_data
        patient.save()
        return video_file_id
    except (DoesNotExist, ValueError, IOError) as e:
        raise ValueError(f"Failed to save speech video: {str(e)}")

def update_speech_excel_reference(patient_id, session_id, excel_file_id, text):
    """Update speech Excel reference and text in session"""
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        patient.sessions[session_index].text = text
        
        feature_data = patient.sessions[session_index].feature_data or {}
        speech_data = feature_data.get("Speech", {})
        speech_data["speech_excel"] = ObjectId(excel_file_id)
        feature_data["Speech"] = speech_data
        
        patient.sessions[session_index].feature_data = feature_data
        patient.save()
        print("Updated session with new speech excel reference")
        return True
    except (DoesNotExist, ValueError) as e:
        raise ValueError(f"Failed to update speech Excel reference: {str(e)}")

def update_speech_plot_reference(patient_id, session_id, plot_file_ids):
    """Update speech plot references in session"""
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        feature_data = patient.sessions[session_index].feature_data or {}
        if "Speech" not in feature_data:
            feature_data["Speech"] = {}
            
        feature_data["Speech"]["plot_images"] = [ObjectId(pid) for pid in plot_file_ids]
        
        patient.sessions[session_index].feature_data = feature_data
        patient.save()
    except (DoesNotExist, ValueError) as e:
        raise ValueError(f"Failed to update speech plot reference: {str(e)}")

def update_speech_report_pdf_reference(patient_id, session_id, pdf_file_id):
    """Update speech report PDF reference in session"""
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        patient.sessions[session_index].report = ObjectId(pdf_file_id)
        patient.save()
        print(f"✓ PDF reference added to patient {patient_id}, session {session_id}")
    except (DoesNotExist, ValueError) as e:
        raise ValueError(f"Failed to update PDF reference: {str(e)}")

# -------------------------------
# File Retrieval Functions (Pure Repository Layer)
# -------------------------------

def get_session_feature_data(patient_id, session_id):
    """
    Get raw feature data from a session (pure data retrieval)
    This function only retrieves data without generating reports
    """
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        session = patient.sessions[session_index]
        
        fer_data = session.feature_data.get("FER", {}) if session.feature_data else {}
        speech_data = session.feature_data.get("Speech", {}) if session.feature_data else {}

        return {
            "session": session,
            "patient": patient,
            "fer_data": fer_data,
            "speech_data": speech_data,
            "session_text": session.text
        }
    except (DoesNotExist, ValueError) as e:
        raise ValueError(f"Failed to get session feature data: {str(e)}")

def get_feature_files_from_session(patient_id, session_id):
    """
    DEPRECATED: This function has been moved to ReportService
    Use ReportService.generate_session_analysis_report() instead
    
    This wrapper is kept for backward compatibility only
    """
    # Import here to avoid circular imports
    from services.report_service import ReportService
    
    print("Warning: get_feature_files_from_session() is deprecated. Use ReportService.generate_session_analysis_report()")
    return ReportService.generate_session_analysis_report(patient_id, session_id)

def download_report_by_id(report_id):
    """Download report by ObjectId"""
    try:
        file = fs.get(ObjectId(report_id))
        return file.filename, file.read(), file.content_type
    except Exception as e:
        print(f"❌ Error fetching report: {e}")
        return None, None, None
    

def update_transcription_pdf_reference(patient_id, session_id, pdf_file_id):
    """Update transcription PDF reference in session.transcription field"""
    try:
        patient = _get_patient_by_id(patient_id)
        session_index = _find_session_index(patient, session_id)
        
        # Store the transcription PDF in the transcription field (not report field)
        patient.sessions[session_index].transcription = ObjectId(pdf_file_id)
        patient.save()
        print(f"✓ Transcription PDF reference added to patient {patient_id}, session {session_id}")
        return True
    except (DoesNotExist, ValueError) as e:
        raise ValueError(f"Failed to update transcription PDF reference: {str(e)}")