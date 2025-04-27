from mongoengine import (
    Document,
    EmbeddedDocument,
    EmbeddedDocumentField,
    StringField,
    IntField,
    ListField,
    DateTimeField,
    DictField,
    ObjectIdField
)

# ------------------------
# Embedded Documents
# ------------------------

class EmergencyContact(EmbeddedDocument):
    name = StringField()
    phone = StringField()
    relation = StringField()

class ContactInformation(EmbeddedDocument):
    email = StringField()
    phone_number = StringField()
    emergency_contact = EmbeddedDocumentField(EmergencyContact, )

class HealthInfo(EmbeddedDocument):
    current_medications = StringField()
    family_history_of_mental_illness = StringField()
    physical_health_conditions = StringField()
    previous_diagnoses = StringField()
    substance_use = StringField()

class TherapyInfo(EmbeddedDocument):
    reason_for_therapy = StringField()

class PersonalInfo(EmbeddedDocument):
    full_name = StringField()
    date_of_birth = StringField()  
    gender = StringField()
    occupation = StringField()
    marital_status = StringField()
    location = StringField()
    contact_information = EmbeddedDocumentField(ContactInformation )
    health_info = EmbeddedDocumentField(HealthInfo )   
    therapy_info = EmbeddedDocumentField(TherapyInfo) 

class Session(EmbeddedDocument):
    session_id = IntField()
    feature_type = StringField()
    date = DateTimeField()
    time = StringField()
    duration = StringField()
    session_type = StringField()
    text = StringField()
    report = ObjectIdField()
    doctor_notes = StringField()
    feature_data = DictField()
    audio_files = ObjectIdField()
    video_files = ObjectIdField()

# ------------------------
# Main Patient Document
# ------------------------

class Patient(Document):
    meta = {'collection': 'Patients'}

    patientID = IntField(required=True, unique=True) 
    personal_info = EmbeddedDocumentField(PersonalInfo)
    registration_date = DateTimeField()
    status = StringField(default="active")
    sessions = ListField(EmbeddedDocumentField(Session))
