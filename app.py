"""
SeranityAI Backend Application
Main Flask application - Clean and Simple
"""

import json
import datetime
import os
import logging
from bson import ObjectId

from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from mongoengine import connect
from flask_swagger_ui import get_swaggerui_blueprint

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set matplotlib backend
os.environ["MPLBACKEND"] = "Agg"

# ================================
# IMPORT CONTROLLERS
# ================================

from controller.doctor_controller import doctor_blueprint
from controller.patient_controller import patient_bp
from controller.file_controller import file_bp
from controller.speech_controller import speech_bp
from controller.chat_controller import chat_bp
from controller.Speech_transcription_controller import speech_recognition_bp
from controller.doctor_notes_controller import doctor_notes_bp


# ================================
# CUSTOM JSON ENCODER
# ================================

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB ObjectId and datetime objects"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

# ================================
# FLASK APP SETUP
# ================================

app = Flask(__name__, static_folder=None)
app.json_encoder = JSONEncoder  
CORS(app)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# MONGODB CONNECTION
# ================================

try:
    connect(db="seranityAI", host="localhost", port=27017, alias="default")
    logger.info("✅ MongoDB connected successfully")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")

# ================================
# SWAGGER UI (OPTIONAL)
# ================================

SWAGGER_URL = "/patientAPI/docs"
API_URL = "/static/swagger.json"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={"app_name": "Patient Management API"}
)

# Uncomment to enable Swagger UI
# app.register_blueprint(swaggerui_blueprint)

# ================================
# REGISTER BLUEPRINTS
# ================================

# Core functionality
app.register_blueprint(doctor_blueprint)
app.register_blueprint(patient_bp)
app.register_blueprint(file_bp)

# Analysis services
app.register_blueprint(speech_bp)
app.register_blueprint(speech_recognition_bp)

# AI services
app.register_blueprint(chat_bp)
app.register_blueprint(doctor_notes_bp, url_prefix='/api/doctor-notes')


logger.info("✅ All blueprints registered successfully")

# ================================
# FLASK-RESTFUL (OPTIONAL)
# ================================

# Uncomment if you need flask-restful
# api = Api(app)
# api.add_resource(YourResource, '/your-resource')

# ================================
# HEALTH CHECK
# ================================

@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint"""
    return {"status": "ok"}, 200

# ================================
# ERROR HANDLERS
# ================================

@app.errorhandler(404)
def not_found(error):
    return {"error": "Endpoint not found"}, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return {"error": "Internal server error"}, 500

# ================================
# RUN SERVER
# ================================

if __name__ == "__main__":
    logger.info("🚀 Starting SeranityAI Backend Server...")
    logger.info("🌐 Server available at: http://0.0.0.0:5001")
    logger.info("📚 Health check: http://0.0.0.0:5001/health")
    
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)