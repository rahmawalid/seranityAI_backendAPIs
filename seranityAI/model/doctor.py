from mongoengine import (
    Document,
    EmbeddedDocument,
    EmbeddedDocumentField,
    StringField,
    IntField,
    ListField,
    DateTimeField,
    EmailField
)
from bson.objectid import ObjectId

# ------------------------
# Embedded Documents
# ------------------------

class ContactInformation(EmbeddedDocument):
    email = EmailField(required=True)
    phone_number = StringField(required=True)

# ------------------------
# Main Doctor Document
# ------------------------

class Doctor(Document):
    meta = {"collection": "Doctors"}

    id = IntField(required=True, unique=True)
    full_name = StringField(required=True)
    date_of_birth = DateTimeField(required=True)
    gender = StringField(required=True)
    specialization = StringField(required=True)
    experience_years = IntField()
    license_number = StringField(required=True)
    contact_information = EmbeddedDocumentField(ContactInformation, required=True)
    profile_picture = StringField()
    password = StringField(required=True)  # i want to hash it
    registration_date = DateTimeField(required=True)
    status = StringField(default="active")  # active / inactive
