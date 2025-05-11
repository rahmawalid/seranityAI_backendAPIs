from mongoengine import (
    Document,
    EmbeddedDocument,
    DynamicEmbeddedDocument,
    StringField,
    EmbeddedDocumentField,
    ObjectIdField,
    IntField,
    ListField,
    EmailField,
    SequenceField,
)
from werkzeug.security import generate_password_hash, check_password_hash


class PersonalInfo(EmbeddedDocument):
    full_name = StringField(required=True)
    date_of_birth = StringField(required=True)
    gender = StringField(required=True)
    email = EmailField(required=True, unique=True)
    phone_number = StringField(
        required=True, unique=True
    )  # stored as string for flexibility
    specialization = StringField(required=True)
    profile_picture = ObjectIdField()  # stored in GridFS


class VerificationDocuments(DynamicEmbeddedDocument):
    medical_license = ObjectIdField()
    degree_certificate = ObjectIdField()
    syndicate_card = ObjectIdField()
    specialization_certificate = ObjectIdField()
    National_ID = ObjectIdField()


class Doctor(Document):
    meta = {"collection": "Doctors"}

    personal_info = EmbeddedDocumentField(PersonalInfo, required=True)
    license_number = StringField(required=True)
    workplace = StringField()
    years_of_experience = StringField()
    # verification_documents = EmbeddedDocumentField(VerificationDocuments)

    password = StringField(required=True)  # storing hashed password
    patientIDs = ListField(IntField())  # list of linked patient IDs
    doctor_ID = SequenceField(
        sequence_name="doctor_id_seq",
        value_decorator=lambda seq: f"D{seq}",
        unique=True,
    )

    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password, method="pbkdf2:sha256")

    def check_password(self, raw_password):
        return check_password_hash(self.password, raw_password)
