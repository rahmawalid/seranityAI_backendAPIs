from flask import Blueprint, jsonify, request
from mongoengine import DoesNotExist, ValidationError
from bson import ObjectId
from model.doctor import Doctor
from repository.doctorRepository import (
    create_doctor,
    get_doctor_by_id,
    update_doctor,
    delete_doctor,
    get_doctor_by_email,
    list_all_doctors
)

doctor_blueprint = Blueprint("doctor_blueprint", __name__)

# -------------------
# Helper
# -------------------
def mongo_to_dict(doc):
    data = doc.to_mongo().to_dict()
    
    # Clean ObjectId and internal fields
    if "_id" in data:
        data["_id"] = str(data["_id"])
    
    if "registration_date" in data and hasattr(data["registration_date"], "isoformat"):
        data["registration_date"] = data["registration_date"].isoformat()
    
    return data

# -------------------
# Create Doctor
# -------------------
@doctor_blueprint.route("/create-doctor", methods=["POST"])
def create_doctor_controller():
    try:
        doctor_data = request.get_json()
        doctor = create_doctor(doctor_data)
        return jsonify({"message": "Doctor created successfully", "doctor_id": doctor.id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Get Doctor by ID
# -------------------
@doctor_blueprint.route("/get-doctor/<int:doctor_id>", methods=["GET"])
def get_doctor_controller(doctor_id):
    try:
        doctor = get_doctor_by_id(doctor_id)
        return jsonify(mongo_to_dict(doctor)), 200
    except (DoesNotExist, ValueError):
        return jsonify({"error": "Doctor not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Update Doctor
# -------------------
@doctor_blueprint.route("/update-doctor/<int:doctor_id>", methods=["PUT"])
def update_doctor_controller(doctor_id):
    try:
        updated_data = request.get_json()
        doctor = update_doctor(doctor_id, updated_data)
        if doctor:
            return jsonify(mongo_to_dict(doctor)), 200
        return jsonify({"error": "Doctor not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Delete Doctor
# -------------------
@doctor_blueprint.route("/delete-doctor/<int:doctor_id>", methods=["DELETE"])
def delete_doctor_controller(doctor_id):
    try:
        delete_doctor(doctor_id)
        return jsonify({"message": "Doctor deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------
# Get Doctor by Email
# -------------------
@doctor_blueprint.route("/get-doctor-by-email", methods=["POST"])
def get_doctor_by_email_controller():
    try:
        email = request.json.get("email")
        doctor = get_doctor_by_email(email)
        return jsonify(mongo_to_dict(doctor)), 200
    except DoesNotExist:
        return jsonify({"error": "Doctor not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# # -------------------
# # List All Doctors
# # -------------------
# @doctor_blueprint.route("/list-doctors", methods=["GET"])
# def list_doctors_controller():
#     try:
#         doctors = list_all_doctors()
#         doctors_list = [mongo_to_dict(d) for d in doctors]
#         return jsonify(doctors_list), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
