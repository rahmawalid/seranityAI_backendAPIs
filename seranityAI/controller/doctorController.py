from flask import Blueprint, request, jsonify, Response
from bson import ObjectId
from flask_cors import cross_origin
from repository.doctorRepository import (
    create_doctor,
    login_doctor,
    update_doctor_password,
    update_doctor,
    get_doctor_by_id,
    delete_doctor,
    save_file_to_gridfs,
    get_verification_file_by_type,
)
from model.doctor import VerificationDocuments
import gridfs
from pymongo import MongoClient
from repository.patientRepository import (
    get_patients_by_doctor,
    create_patient_for_doctor,
)
from controller.patientController import _cors_preflight, mongo_to_dict_patient


doctor_blueprint = Blueprint("doctor_blueprint", __name__)

mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

VALID_DOC_FIELDS = {
    "medical_license",
    "degree_certificate",
    "syndicate_card",
    "specialization_certificate",
    "national_id",
}


# ---------------------------
# Helper
# ---------------------------
def mongo_to_dict(doc):
    data = doc.to_mongo().to_dict()
    data["_id"] = str(data["_id"])
    if "personal_info" in data and data["personal_info"].get("profile_picture"):
        data["personal_info"]["profile_picture"] = str(
            data["personal_info"]["profile_picture"]
        )
    # if "verification_documents" in data:
    #     for k, v in data["verification_documents"].items():
    #         if v:
    #             data["verification_documents"][k] = str(v)
    return data


# ---------------------------
# Create Doctor
# ---------------------------
@doctor_blueprint.route("/create-doctor", methods=["POST"])
def create_doctor_controller():
    try:
        doctor_data = request.get_json()
        if not doctor_data:
            return jsonify({"error": "Invalid or missing JSON data"}), 400

        doctor = create_doctor(doctor_data)
        return (
            jsonify({"message": "Doctor created", "doctor_ID": doctor.doctor_ID}),
            201,
        )

    except ValueError as ve:
        print(ve)
        return jsonify({"error": str(ve)}), 409  # Conflict
    except Exception as e:
        return jsonify({"error": "Server error: " + str(e)}), 500


# ---------------------------
# Login Doctor
# ---------------------------
@doctor_blueprint.route("/login-doctor", methods=["POST"])
def login_doctor_controller():
    try:
        data = request.get_json()
        doctor = login_doctor(data["email"], data["password"])
        print(doctor)
        return (
            jsonify(
                {
                    "message": "Login successful",
                    "doctor_ID": doctor.doctor_ID,
                    "doctor_info": mongo_to_dict(doctor),
                }
            ),
            201,
        )
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 401


# ---------------------------
# Update Doctor Password
# ---------------------------
@doctor_blueprint.route("/update-doctor-password/<doctor_ID>", methods=["PUT"])
def update_password_controller(doctor_ID):
    try:
        data = request.get_json()
        old_password = data.get("old_password")
        new_password = data.get("new_password")

        if not all([old_password, new_password]):
            return jsonify({"error": "Both old and new passwords are required"}), 400

        update_doctor_password(doctor_ID, old_password, new_password)
        return jsonify({"message": "Password updated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------
# Update Doctor Info
# ---------------------------
@doctor_blueprint.route("/update-doctor-info/<doctor_ID>", methods=["PUT"])
def update_doctor_info_controller(doctor_ID):
    try:
        update_data = request.get_json()
        updated = update_doctor(doctor_ID, update_data)
        if updated:
            return jsonify(mongo_to_dict(updated)), 200
        return jsonify({"error": "Doctor not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Get Doctor
# ---------------------------
@doctor_blueprint.route("/get-doctor/<doctor_ID>", methods=["GET"])
def get_doctor_controller(doctor_ID):
    try:
        doctor = get_doctor_by_id(doctor_ID)
        return jsonify(mongo_to_dict(doctor)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


# ---------------------------
# Delete Doctor
# ---------------------------
@doctor_blueprint.route("/delete-doctor/<doctor_ID>", methods=["DELETE"])
def delete_doctor_controller(doctor_ID):
    try:
        delete_doctor(doctor_ID)
        return jsonify({"message": "Doctor deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Upload Doctor File
# ---------------------------
@doctor_blueprint.route("/upload-doctor-file/<file_type>/<doctor_ID>", methods=["POST"])
def upload_doctor_file(file_type, doctor_ID):
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        if file_type != "profile_picture" and file_type not in VALID_DOC_FIELDS:
            return jsonify({"error": f"Invalid file type: {file_type}"}), 400

        file = request.files["file"]
        content_type = file.content_type
        file_id = save_file_to_gridfs(file, content_type)

        doctor = get_doctor_by_id(doctor_ID)

        if file_type == "profile_picture":
            doctor.personal_info.profile_picture = ObjectId(file_id)
        else:
            if not doctor.verification_documents:
                doctor.verification_documents = VerificationDocuments()
            setattr(doctor.verification_documents, file_type, ObjectId(file_id))

        doctor.save()
        return jsonify({"message": "File uploaded", "file_id": file_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Get Verification File
# ---------------------------
@doctor_blueprint.route(
    "/get-verification-file/<file_type>/<doctor_ID>", methods=["GET"]
)
def get_verification_file(file_type, doctor_ID):
    try:
        if file_type not in VALID_DOC_FIELDS:
            return jsonify({"error": "Invalid verification document type"}), 400

        file = get_verification_file_by_type(doctor_ID, file_type)
        return Response(file.read(), mimetype=file.content_type)

    except Exception as e:
        return jsonify({"error": str(e)}), 404


@doctor_blueprint.route("/<doctor_ID>/patients", methods=["OPTIONS", "GET"])
@cross_origin()
def list_patients_for_doctor(doctor_ID):
    if request.method == "OPTIONS":
        return _cors_preflight()
    try:
        patients = get_patients_by_doctor(doctor_ID)
        out = [mongo_to_dict_patient(p) for p in patients]
        return jsonify(out), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@doctor_blueprint.route("/<doctor_ID>/patients", methods=["OPTIONS", "POST"])
@cross_origin()
def add_patient_for_doctor(doctor_ID):
    if request.method == "OPTIONS":
        return _cors_preflight()
    data = request.get_json() or {}
    try:
        p = create_patient_for_doctor(doctor_ID, data)
        return (
            jsonify(
                {
                    "message": "Patient created",
                    "patientID": p.patientID,
                    "doctorID": p.doctorID,
                }
            ),
            201,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400
