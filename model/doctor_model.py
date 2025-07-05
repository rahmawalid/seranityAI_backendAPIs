from mongoengine import (
    Document,
    EmbeddedDocument,
    StringField,
    EmbeddedDocumentField,
    ObjectIdField,
    IntField,
    ListField,
    DateTimeField,
    EmailField,
    SequenceField,
    BooleanField,
)
from werkzeug.security import generate_password_hash, check_password_hash


class PersonalInfo(EmbeddedDocument):
    # From signup form - Step 1
    full_name = StringField(required=True)
    date_of_birth = StringField(required=True)
    gender = StringField(required=True)
    email = EmailField(required=True, unique=True)
    phone_number = StringField(required=True, unique=True)
    
    # From signup form - Step 2 (Professional Info)
    specialization = StringField(required=True)
    
    # Optional field for profile picture (can be added later)
    profile_picture = ObjectIdField()


class ScheduledSession(EmbeddedDocument):
    patientID = IntField(db_field="patientID", required=True)
    datetime = DateTimeField(db_field="datetime", required=True)
    notes = StringField(db_field="notes")


class Doctor(Document):
    meta = {"collection": "Doctors"}

    # Core personal information from signup form
    personal_info = EmbeddedDocumentField(PersonalInfo, required=True)
    
    # Professional information from signup form - Step 2
    license_number = StringField(required=True)
    workplace = StringField()  # Optional - "Clinic, Hospital, online etc."
    years_of_experience = StringField()  # Optional - "How many years of experience do you have?"
    
    # Password from signup form - Step 2
    password = StringField(required=True)
    
    # Email verification system
    email_verified = BooleanField(default=False)
    verification_token = StringField()
    token_created_at = DateTimeField()
    
    # System generated fields - FIXED: Use StringField instead of SequenceField
    doctor_ID = StringField(
        db_field="doctor_ID",
        unique=True,
    )
    
    # Patient and session management
    patientIDs = ListField(IntField())
    scheduledSessions = ListField(
        EmbeddedDocumentField(ScheduledSession),
        db_field="scheduledSessions",
        required=False,
        default=[],
    )

    def set_password(self, raw_password):
        """Hash and set the doctor's password"""
        if raw_password:
            self.password = generate_password_hash(raw_password, method="pbkdf2:sha256")

    def check_password(self, raw_password):
        """Check if the provided password matches the stored hash"""
        if not self.password or not raw_password:
            return False
        return check_password_hash(self.password, raw_password)
    
    def save(self, *args, **kwargs):
        """Override save to generate doctor_ID if not present"""
        if not self.doctor_ID:
            # Get the highest existing doctor ID number
            from mongoengine import Q
            last_doctor = Doctor.objects(
                doctor_ID__regex=r'^D\d+$'
            ).order_by('-doctor_ID').first()
            
            if last_doctor and last_doctor.doctor_ID:
                try:
                    last_num = int(last_doctor.doctor_ID[1:])
                    next_num = last_num + 1
                except (ValueError, IndexError):
                    next_num = 1
            else:
                next_num = 1
            
            self.doctor_ID = f"D{next_num}"
        
        return super().save(*args, **kwargs)