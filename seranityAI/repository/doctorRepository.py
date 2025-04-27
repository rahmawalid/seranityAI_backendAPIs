
import gridfs
from bson import ObjectId
from pymongo import MongoClient
from mongoengine import DoesNotExist, ValidationError
from model.doctor import Doctor

# doctor profile pictures or docs
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

# -------------------------------
# Doctor CRUD Operations
# -------------------------------

def create_doctor(doctor_data):
    try:
        doctor = Doctor(**doctor_data)
        doctor.save()
        return doctor
    except Exception as e:
        raise e

def get_doctor_by_id(doctor_id):
    try:
        return Doctor.objects.get(id=int(doctor_id))
    except (DoesNotExist, ValidationError, ValueError) as e:
        raise e

def update_doctor(doctor_id, doctor_data):
    try:
        return Doctor.objects(id=int(doctor_id)).modify(new=True, **doctor_data)
    except (DoesNotExist, ValidationError) as e:
        raise e

def delete_doctor(doctor_id):
    try:
        Doctor.objects(id=int(doctor_id)).delete()
    except ValidationError as e:
        raise e

def get_doctor_by_email(email):
    try:
        return Doctor.objects.get(contact_information__email=email)
    except DoesNotExist as e:
        raise e

def list_all_doctors():
    try:
        return Doctor.objects()
    except Exception as e:
        raise e
