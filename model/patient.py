from mongoengine import (
    Document,
    EmbeddedDocument,
    DynamicEmbeddedDocument,
    EmbeddedDocumentField,
    StringField,
    IntField,
    ListField,
    DateTimeField,
    DictField,
    ObjectIdField,
)


# ------------------------
# Embedded Documents
# ------------------------
class EmergencyContact(DynamicEmbeddedDocument):
    name = StringField(db_field="name")
    phone = StringField(db_field="phone")
    relation = StringField(db_field="relation")


class ContactInformation(DynamicEmbeddedDocument):
    email = StringField(db_field="email")
    phone_number = StringField(db_field="phone_number")
    emergency_contact = EmbeddedDocumentField(
        EmergencyContact, db_field="emergency_contact"
    )


class HealthInfo(DynamicEmbeddedDocument):
    current_medications = StringField(db_field="current_medications")
    family_history_of_mental_illness = StringField(
        db_field="family_history_of_mental_illness"
    )
    physical_health_conditions = StringField(db_field="physical_health_conditions")
    previous_diagnoses = StringField(db_field="previous_diagnoses")
    substance_use = StringField(db_field="substance_use")


class TherapyInfo(DynamicEmbeddedDocument):
    reason_for_therapy = StringField(db_field="reason_for_therapy")


class PersonalInfo(DynamicEmbeddedDocument):
    full_name = StringField(db_field="fullName")
    date_of_birth = StringField(db_field="dateOfBirth")
    gender = StringField(db_field="gender")
    occupation = StringField(db_field="occupation")
    marital_status = StringField(db_field="marital_status")
    location = StringField(db_field="location")
    contact_information = EmbeddedDocumentField(
        ContactInformation, db_field="contact_information"
    )
    health_info = EmbeddedDocumentField(HealthInfo, db_field="health_info")
    therapy_info = EmbeddedDocumentField(TherapyInfo, db_field="therapy_info")


class Session(DynamicEmbeddedDocument):
    session_id = IntField(db_field="session_id", required=True, unique=True)
    feature_type = StringField(db_field="featureType")
    date = DateTimeField(db_field="date")
    time = StringField(db_field="time")
    duration = StringField(db_field="duration")
    session_type = StringField(db_field="sessionType")
    text = StringField(db_field="text")
    report = StringField(db_field="report")
    doctor_notes = StringField(db_field="doctorNotes")
    feature_data = DictField(db_field="featureData")
    audio_files = ObjectIdField(db_field="audioFiles")
    video_files = ObjectIdField(db_field="videoFiles")
    model_files = DictField(field=ObjectIdField(), db_field="model_files")


# ------------------------
# Main Patient Document
# ------------------------
class Patient(Document):
    meta = {
        "collection": "Patients",
        "indexes": [
            "patientID",
        ],
    }

    # existing patient ID
    patientID = IntField(db_field="patientID", required=True, unique=True)
    # new doctor ID (e.g. "D1", "D2", etc.)
    doctorID = StringField(db_field="doctorID")

    personal_info = EmbeddedDocumentField(
        PersonalInfo, db_field="personalInfo", required=True
    )
    registration_date = DateTimeField(db_field="registration_date")
    status = StringField(db_field="status", default="active")
    sessions = ListField(EmbeddedDocumentField(Session), db_field="sessions")
