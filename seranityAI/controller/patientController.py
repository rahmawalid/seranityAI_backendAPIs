from flask import Blueprint, Response, jsonify, request
from mongoengine import DoesNotExist, ValidationError
from bson import ObjectId
from model.patient import Patient
from repository.patientRepository import (
    create_patient,
    save_audio_to_gridfs,
    attach_audio_to_session,
    save_video_to_gridfs,
    attach_video_to_session,
    save_pdf_to_gridfs,
    attach_pdf_to_session,
    update_patient,
    delete_patient,
    get_patient_by_id,
    get_patient_by_email
)
from pymongo import MongoClient
from repository.FER import get_video_file_in_memory

import gridfs

mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

patient_blueprint = Blueprint("patient_blueprint", __name__)

# -------------------
# Helper
# -------------------
def mongo_to_dict(doc):
    data = doc.to_mongo().to_dict()
    
    # Convert internal Mongo _id to string
    if '_id' in data:
        data['_id'] = str(data['_id'])
    
    # Convert ObjectId fields inside sessions
    if "sessions" in data:
        for session in data["sessions"]:
            if session.get("audio_files"):
                session["audio_files"] = str(session["audio_files"])
            if session.get("video_files"):
                session["video_files"] = str(session["video_files"])
            if session.get("report"):
                session["report"] = str(session["report"])
    
    return data

# -------------------
# Create Patient
# -------------------
@patient_blueprint.route("/create-patient", methods=["POST"])
def create_patient_controller():
    try:
        patient_data = request.get_json()

        created_patient = create_patient(patient_data)

        return jsonify({
            "message": "Patient created successfully",
            "patient_id": created_patient.patientID
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Get Patient
# -------------------
@patient_blueprint.route("/get-patient/<int:patient_id>", methods=["GET"])
def get_patient_controller(patient_id):
    try:
        patient = get_patient_by_id(patient_id)
        return jsonify(mongo_to_dict(patient)), 200
    except (DoesNotExist, ValueError):
        return jsonify({"error": "Patient not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Update Patient
# -------------------
@patient_blueprint.route("/update-patient/<int:patient_id>", methods=["PUT"])
def update_patient_controller(patient_id):
    try:
        updated_data = request.get_json()
        patient = update_patient(patient_id, updated_data)
        if patient:
            return jsonify(mongo_to_dict(patient)), 200
        return jsonify({"error": "Patient not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Delete Patient
# -------------------
@patient_blueprint.route("/delete-patient/<int:patient_id>", methods=["DELETE"])
def delete_patient_controller(patient_id):
    try:
        delete_patient(patient_id)
        return jsonify({"message": "Patient deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# List All Patients
# -------------------
@patient_blueprint.route("/list-patients", methods=["GET"])
def list_patients_controller():
    try:
        patients = Patient.objects()
        patients_list = [mongo_to_dict(p) for p in patients]
        return jsonify(patients_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Upload Audio, Video, Report
# -------------------

@patient_blueprint.route("/upload-audio/<int:patient_id>/<int:session_id>", methods=["POST"])
def upload_audio_file(patient_id, session_id):
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if file.filename.lower().endswith(".mp3"):
            file_id = save_audio_to_gridfs(file)
            if attach_audio_to_session(patient_id, session_id, file_id):
                return jsonify({"message": "Audio uploaded", "file_id": file_id}), 200
            return jsonify({"error": "Session not found"}), 404
        return jsonify({"error": "Only .mp3 files allowed"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@patient_blueprint.route("/upload-video/<int:patient_id>/<int:session_id>", methods=["POST"])
def upload_video_file(patient_id, session_id):
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if file.filename.lower().endswith(".mp4"):
            file_id = save_video_to_gridfs(file)
            if attach_video_to_session(patient_id, session_id, file_id):
                return jsonify({"message": "Video uploaded", "file_id": file_id}), 200
            return jsonify({"error": "Session not found"}), 404
        return jsonify({"error": "Only .mp4 files allowed"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@patient_blueprint.route("/upload-report/<int:patient_id>/<int:session_id>", methods=["POST"])
def upload_report_file(patient_id, session_id):
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if file.filename.lower().endswith(".pdf"):
            file_id = save_pdf_to_gridfs(file)
            if attach_pdf_to_session(patient_id, session_id, file_id):
                return jsonify({"message": "PDF uploaded", "file_id": file_id}), 200
            return jsonify({"error": "Session not found"}), 404
        return jsonify({"error": "Only .pdf files allowed"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Stream Audio / Video / PDF
# -------------------

@patient_blueprint.route("/stream-audio/<file_id>", methods=["GET"])
def stream_audio(file_id):
    try:
        file = fs.get(ObjectId(file_id))
        return Response(file.read(), mimetype="audio/mp3")
    except Exception as e:
        return jsonify({"error": f"Audio not found: {str(e)}"}), 404

@patient_blueprint.route("/stream-video/<file_id>", methods=["GET"])
def stream_video(file_id):
    try:
        file = fs.get(ObjectId(file_id))
        return Response(file.read(), mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": f"Video not found: {str(e)}"}), 404

@patient_blueprint.route("/view-report/<file_id>", methods=["GET"])
def view_report(file_id):
    try:
        file = fs.get(ObjectId(file_id))
        return Response(file.read(), mimetype="application/pdf")
    except Exception as e:
        return jsonify({"error": f"PDF not found: {str(e)}"}), 404


@patient_blueprint.route("/stream-video-in-memory/<file_id>", methods=["GET"])
def stream_video_in_memory(file_id):
    try:
        video_file = get_video_file_in_memory(file_id)
        return Response(video_file, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


