from flask import Flask
from flask_restful import Api
from flask_restful_swagger import swagger
from mongoengine import connect  # ✅ Import connect here
from config import Config
from controller.patientController import patient_blueprint
from flask_swagger_ui import get_swaggerui_blueprint

from controller.doctorController import doctor_blueprint

# ------------------
# App Initialization
# ------------------
app = Flask(__name__)
app.config.from_object(Config)

# ------------------
# MongoDB Connection
# ------------------
connect(**app.config["MONGODB_SETTINGS"])  
# ------------------
# Swagger UI Setup
# ------------------
SWAGGER_URL = "/patientAPI/docs"
API_URL = "/static/swagger.json"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={"app_name": "Patient Management API"}
)

# ------------------
# Register Blueprints
# ------------------
app.register_blueprint(swaggerui_blueprint)
app.register_blueprint(patient_blueprint)

app.register_blueprint(doctor_blueprint)

# ------------------
# RESTful API
# ------------------
api = Api(app)

# ------------------
# Run the App
# ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)


