import gridfs
from bson import ObjectId
from pymongo import MongoClient
from mongoengine import DoesNotExist, ValidationError
import os
import time
import pandas as pd
import cv2
import numpy as np
import wave
import matplotlib.pyplot as plt
from fer import Video, FER
from moviepy.editor import VideoFileClip
from moviepy.editor import VideoFileClip
from model.patient import Patient

from mongoengine import NotUniqueError
import io

# -------------------------------
# MongoDB Setup (GridFS)
# -------------------------------
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

# -------------------------------
# Patient CRUD Operations
# -------------------------------


def get_patient_by_id(patient_id):
    try:
        return Patient.objects.get(patientID=int(patient_id))
    except (DoesNotExist, ValidationError, ValueError) as e:
        raise e


def create_patient(patient_data):
    try:
        patient = Patient(**patient_data)
        patient.save()
        return patient
    except NotUniqueError as e:
        raise ValueError("Patient ID must be unique!")  # custom friendly message
    except (ValidationError, Exception) as e:
        raise e


def update_patient(patient_id, patient_data):
    try:
        return Patient.objects(patientID=int(patient_id)).modify(
            new=True, **patient_data
        )
    except (DoesNotExist, ValidationError) as e:
        raise e


def delete_patient(patient_id):
    try:
        Patient.objects(patientID=int(patient_id)).delete()
    except ValidationError as e:
        raise e


def get_patient_by_email(email):
    try:
        return Patient.objects.get(personal_info__contact_information__email=email)
    except DoesNotExist as e:
        raise e


# -------------------------------
# Audio / Video / PDF / Model‑File helpers
# -------------------------------
def save_audio_to_gridfs(file_storage):
    return str(
        fs.put(file_storage, filename=file_storage.filename, content_type="audio/mp3")
    )


def attach_audio_to_session(patient_id, session_id, file_id):
    patient = Patient.objects.get(patientID=int(patient_id))
    print("Patient ID:", patient_id)
    print("Session ID:", session_id)
    print("Patient Object:", patient)
    for session in patient.sessions:
        if session.session_id == session_id:
            session.audio_files = ObjectId(file_id)
            patient.save()
            return True
    return False


def save_video_to_gridfs(file_storage):
    return str(
        fs.put(file_storage, filename=file_storage.filename, content_type="video/mp4")
    )


def attach_video_to_session(patient_id, session_id, file_id):
    patient = Patient.objects.get(patientID=int(patient_id))
    for session in patient.sessions:
        if session.session_id == session_id:
            session.video_files = ObjectId(file_id)
            patient.save()
            return True
    return False


def save_pdf_to_gridfs(file_storage):
    return str(
        fs.put(
            file_storage, filename=file_storage.filename, content_type="application/pdf"
        )
    )


def attach_pdf_to_session(patient_id, session_id, file_id):
    patient = Patient.objects.get(patientID=int(patient_id))
    for session in patient.sessions:
        if session.session_id == session_id:
            session.report = ObjectId(file_id)
            patient.save()
            return True
    return False


def save_excel_to_gridfs(file_storage):
    return str(
        fs.put(
            file_storage,
            filename=file_storage.filename,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    )


def attach_model_file(patient_id, session_id, model_type, file_label, file_id):
    patient = Patient.objects.get(patientID=int(patient_id))
    for session in patient.sessions:
        if session.session_id == session_id:
            session.model_files.setdefault(model_type, {})[file_label] = ObjectId(
                file_id
            )
            patient.save()
            return True
    return False


def get_patients_by_doctor(doctor_id: str):
    """
    Return a QuerySet of all Patient documents whose doctorID equals the given string.
    """
    try:
        return Patient.objects(doctorID=doctor_id)
    except (DoesNotExist, ValidationError):
        return Patient.objects.none()


def create_patient_for_doctor(doctor_id: str, patient_data: dict):
    """
    Create a new Patient, ensuring its doctorID field is set.
    Uses the existing Patient creation logic (assumes create_patient exists).
    """
    patient_data["doctorID"] = doctor_id

    return create_patient(patient_data)
