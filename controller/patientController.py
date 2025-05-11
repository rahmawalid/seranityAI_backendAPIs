import datetime
from flask import Blueprint, request, jsonify, make_response, Response
from flask_cors import cross_origin
from bson import ObjectId
from mongoengine import DoesNotExist, ValidationError
from pymongo import MongoClient
from repository.patientRepository import (
    create_patient,
    get_patient_by_id,
    update_patient,
    delete_patient,
    save_audio_to_gridfs,
    attach_audio_to_session,
    save_video_to_gridfs,
    attach_video_to_session,
    save_pdf_to_gridfs,
    attach_pdf_to_session,
    save_excel_to_gridfs,
    attach_model_file,
)
from model.patient import Patient

patient_blueprint = Blueprint("patient", __name__, url_prefix="/patient")


def _cors_preflight():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


import gridfs
from bson import ObjectId


mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

patient_blueprint = Blueprint("patient_blueprint", __name__)


# -------------------
# Helper
# -------------------
def mongo_to_dict_patient(doc):
    data = doc.to_mongo().to_dict()

    # top‑level _id
    data["_id"] = str(data["_id"])

    # sessions
    sessions = data.get("sessions", [])
    for sess in sessions:

        d = sess.get("date")
        if isinstance(d, datetime.datetime):
            sess["date"] = d.isoformat()

        for oid_key in ("audioFiles", "videoFiles"):
            if oid_key in sess and isinstance(sess[oid_key], ObjectId):
                sess[oid_key] = str(sess[oid_key])

        mf = sess.get("model_files")
        if isinstance(mf, dict):
            for label, val in mf.items():
                if isinstance(val, ObjectId):
                    mf[label] = str(val)

    return data


# -------------------
# Create Patient
# -------------------
@patient_blueprint.route("/create-patient", methods=["POST"])
def create_patient_controller():
    try:
        patient_data = request.get_json()

        created_patient = create_patient(patient_data)

        return (
            jsonify(
                {
                    "message": "Patient created successfully",
                    "patient_id": created_patient.patientID,
                }
            ),
            201,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------
# Get Patient
# -------------------
@patient_blueprint.route("/get-patient/<int:patient_id>", methods=["GET"])
def get_patient_controller(patient_id):
    try:
        patient = get_patient_by_id(patient_id)
        return jsonify(mongo_to_dict_patient(patient)), 200
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
            return jsonify(mongo_to_dict_patient(patient)), 200
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
        patients_list = [mongo_to_dict_patient(p) for p in patients]
        return jsonify(patients_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Upload Audio ──────────────────────────────────────────────────────────────
@patient_blueprint.route(
    "/upload-audio/<int:patient_id>/<int:session_id>", methods=["OPTIONS", "POST"]
)
@cross_origin()
def upload_audio_file(patient_id, session_id):
    if request.method == "OPTIONS":
        return _cors_preflight()
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify(error="No audio file"), 400
    if not file.filename.lower().endswith((".mp3", ".wav")):
        return jsonify(error="Only .mp3/.wav allowed"), 400
    print("File name:", file.filename)
    file_id = save_audio_to_gridfs(file)
    print("File ID:", file_id)
    if attach_audio_to_session(patient_id, session_id, file_id):
        return jsonify(message="Audio uploaded", file_id=file_id), 200
    return jsonify(error="Session not found"), 404


# ─── Upload Video ──────────────────────────────────────────────────────────────
@patient_blueprint.route(
    "/upload-video/<int:patient_id>/<int:session_id>", methods=["OPTIONS", "POST"]
)
@cross_origin()
def upload_video_file(patient_id, session_id):
    if request.method == "OPTIONS":
        return _cors_preflight()
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify(error="No video file"), 400
    if not file.filename.lower().endswith((".mp4", ".mov")):
        return jsonify(error="Only .mp4/.mov allowed"), 400
    file_id = save_video_to_gridfs(file)
    if attach_video_to_session(patient_id, session_id, file_id):
        return jsonify(message="Video uploaded", file_id=file_id), 200
    return jsonify(error="Session not found"), 404


# ─── Upload PDF ────────────────────────────────────────────────────────────────
@patient_blueprint.route(
    "/upload-report/<int:patient_id>/<int:session_id>", methods=["OPTIONS", "POST"]
)
@cross_origin()
def upload_report_file(patient_id, session_id):
    if request.method == "OPTIONS":
        return _cors_preflight()
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify(error="No PDF"), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify(error="Only .pdf allowed"), 400
    file_id = save_pdf_to_gridfs(file)
    if attach_pdf_to_session(patient_id, session_id, file_id):
        return jsonify(message="PDF uploaded", file_id=file_id), 200
    return jsonify(error="Session not found"), 404


# ─── Upload Model‑File (Excel etc.) ───────────────────────────────────────────
@patient_blueprint.route(
    "/upload-model-file/<int:patient_id>/<int:session_id>/<model_type>/<file_label>",
    methods=["OPTIONS", "POST"],
)
@cross_origin()
def upload_model_file(patient_id, session_id, model_type, file_label):
    if request.method == "OPTIONS":
        return _cors_preflight()
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify(error="No file"), 400
    file_id = save_excel_to_gridfs(file)
    if attach_model_file(patient_id, session_id, model_type, file_label, file_id):
        return jsonify(message="Model file uploaded", file_id=file_id), 200
    return jsonify(error="Session not found"), 404


# ─── Stream Audio ─────────────────────────────────────────────────────────────
@patient_blueprint.route("/stream-audio/<file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def stream_audio(file_id):
    if request.method == "OPTIONS":
        return _cors_preflight()
    try:
        data = fs.get(ObjectId(file_id)).read()
        return Response(data, mimetype="audio/mpeg")
    except:
        return jsonify(error="Audio not found"), 404


# ─── Stream Video (direct & in‑memory) ─────────────────────────────────────────
@patient_blueprint.route("/stream-video/<file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def stream_video(file_id):
    if request.method == "OPTIONS":
        return _cors_preflight()
    try:
        data = fs.get(ObjectId(file_id)).read()
        return Response(data, mimetype="video/mp4")
    except:
        return jsonify(error="Video not found"), 404


# @patient_blueprint.route(
#     "/stream-video-in-memory/<file_id>", methods=["GET", "OPTIONS"]
# )
# @cross_origin()
# def stream_video_in_memory(file_id):
#     if request.method == "OPTIONS":
#         return _cors_preflight()
#     try:
#         data = get_video_file_in_memory(file_id)
#         return Response(data, mimetype="video/mp4")
#     except Exception as e:
#         return jsonify(error=str(e)), 500


# ─── View PDF ─────────────────────────────────────────────────────────────────
@patient_blueprint.route("/view-report/<file_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def view_report(file_id):
    if request.method == "OPTIONS":
        return _cors_preflight()
    try:
        data = fs.get(ObjectId(file_id)).read()
        return Response(data, mimetype="application/pdf")
    except:
        return jsonify(error="PDF not found"), 404
