"""
Test script to verify the doctor query fix
"""

from mongoengine import connect
from model.doctor_model import Doctor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_doctor_fix():
    """Test the doctor query fix"""
    try:
        # Connect to database
        connect(db="seranityAI", host="localhost", port=27017, alias="default")
        
        logger.info("🔍 Testing Doctor Query Fix")
        
        # Test 1: Raw query (should work)
        logger.info("\n1. Testing raw query...")
        doctor_raw = Doctor.objects(__raw__={"doctor_ID": "D15"}).first()
        if doctor_raw:
            logger.info("✅ Raw query works!")
            logger.info(f"   Email: {doctor_raw.personal_info.email if doctor_raw.personal_info else 'No email'}")
        else:
            logger.error("❌ Raw query failed")
        
        # Test 2: Standard query (this was failing before)
        logger.info("\n2. Testing standard query...")
        try:
            doctor_std = Doctor.objects(doctor_ID="D15").first()
            if doctor_std:
                logger.info("✅ Standard query now works!")
            else:
                logger.error("❌ Standard query still fails")
        except Exception as e:
            logger.error(f"❌ Standard query error: {e}")
        
        # Test 3: Test the repository function
        logger.info("\n3. Testing get_doctor_by_id function...")
        try:
            from repository.doctor_repository import get_doctor_by_id
            doctor_repo = get_doctor_by_id("D15")
            logger.info("✅ get_doctor_by_id function works!")
            logger.info(f"   Email: {doctor_repo.personal_info.email if doctor_repo.personal_info else 'No email'}")
            
            # Test scheduling function
            logger.info("\n4. Testing scheduling function...")
            from repository.doctor_repository import schedule_session_for_doctor
            import datetime
            
            # Try to schedule a session
            when_iso = (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat()
            session = schedule_session_for_doctor("D15", 1, when_iso, "Test session")
            
            logger.info("✅ Scheduling function works!")
            logger.info(f"   Session scheduled for: {session.datetime}")
            
        except Exception as e:
            logger.error(f"❌ Repository function error: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_doctor_fix()