import gridfs
from bson import ObjectId
from pymongo import MongoClient
from mongoengine import DoesNotExist, ValidationError, NotUniqueError
from model.doctor import Doctor
import io

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)


# ------------------------------
# File Upload to GridFS
# ------------------------------
def save_file_to_gridfs(file_storage, content_type):
    """
    Save a file to GridFS and return its ObjectId as a string.
    """
    return str(
        fs.put(file_storage, filename=file_storage.filename, content_type=content_type)
    )


# ------------------------------
# Create Doctor
# ------------------------------
def create_doctor(doctor_data):
    try:
        if not doctor_data or "password" not in doctor_data:
            raise ValueError("Password is required.")

        email = doctor_data["personal_info"]["email"]
        phone = doctor_data["personal_info"]["phone_number"]

        # Check for uniqueness manually

        if Doctor.objects(personal_info__email=email).first():
            raise ValueError(
                "Email already exists, please login or use a different email."
            )
        if Doctor.objects(personal_info__phone_number=phone).first():
            raise ValueError(
                "Phone number already exists, please use a different number."
            )

        raw_password = doctor_data.pop("password")
        doctor = Doctor(**doctor_data)
        doctor.set_password(raw_password)
        doctor.save()
        return doctor

    except Exception as e:
        raise e


# ------------------------------
# Get Doctor by Key
# ------------------------------
def get_doctor_by_key(doctor_ID):
    try:
        return Doctor.objects.get(doctor_ID=doctor_ID)
    except (DoesNotExist, ValidationError) as e:
        raise e


# ------------------------------
# Update Doctor
# ------------------------------
def update_doctor(doctor_ID, update_data):
    """
    Updates doctor fields by key. Handles nested fields and list updates.
    """
    try:
        if "password" in update_data:
            # Handle password hash
            raw_password = update_data.pop("password")
            doctor = Doctor.objects.get(doctor_ID=doctor_ID)
            doctor.set_password(raw_password)
            doctor.save()
            return doctor

        return Doctor.objects(doctor_ID=doctor_ID).modify(new=True, **update_data)
    except Exception as e:
        raise e


# ------------------------------
# Delete Doctor
# ------------------------------
def delete_doctor(doctor_ID):
    try:
        Doctor.objects(doctor_ID=doctor_ID).delete()
    except Exception as e:
        raise e


# -------------------------------
# Login (Check Password)
# -------------------------------
def login_doctor(email, password):
    try:
        doctor = Doctor.objects.get(personal_info__email=email)
        if doctor.check_password(password):
            return doctor
        else:
            raise ValueError("Invalid credentials.")
    except DoesNotExist:
        raise ValueError("Doctor not found.")


# -------------------------------
# Update Password
# -------------------------------
def update_doctor_password(doctor_ID, old_password, new_password):
    doctor = Doctor.objects.get(doctor_ID=doctor_ID)
    if not doctor.check_password(old_password):
        raise ValueError("Old password is incorrect.")
    doctor.set_password(new_password)
    doctor.save()
    return True


# -------------------------------
# Update Doctor Profile Info
# -------------------------------
def update_doctor(doctor_ID, update_data):
    try:
        return Doctor.objects(doctor_ID=doctor_ID).modify(new=True, **update_data)
    except Exception as e:
        raise e


# -------------------------------
# Get Doctor by doctor_ID
# -------------------------------
def get_doctor_by_id(doctor_ID):
    try:
        return Doctor.objects.get(doctor_ID=doctor_ID)
    except (DoesNotExist, ValidationError) as e:
        raise e


# -------------------------------
# Get Verification File by Type
# -------------------------------
def get_verification_file_by_type(doctor_ID, file_type):
    doctor = get_doctor_by_id(doctor_ID)
    file_id = getattr(doctor.verification_documents, file_type, None)
    if not file_id:
        raise ValueError(f"{file_type} not found.")
    return fs.get(file_id)
