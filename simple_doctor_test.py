"""
Simple test script to isolate the doctor query issue
"""

from mongoengine import connect
from model.doctor_model import Doctor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_doctor_test():
    """Simple test to find the issue"""
    try:
        # Connect to database
        connect(db="seranityAI", host="localhost", port=27017, alias="default")
        
        logger.info("🔍 Simple Doctor Query Test")
        
        # Test 1: Find doctor by email first
        logger.info("\n1. Finding doctor by email...")
        doctor = Doctor.objects(personal_info__email="rahmawshm@gmail.com").first()
        
        if doctor:
            actual_id = doctor.doctor_ID
            logger.info(f"✅ Found doctor with ID: '{actual_id}'")
            logger.info(f"   ID type: {type(actual_id)}")
            logger.info(f"   ID length: {len(actual_id)}")
            logger.info(f"   ID repr: {repr(actual_id)}")
            logger.info(f"   ID bytes: {actual_id.encode('utf-8')}")
            
            # Test 2: Try to find this exact doctor by the ID we just got
            logger.info(f"\n2. Trying to find doctor by ID '{actual_id}'...")
            
            test_doctor = Doctor.objects(doctor_ID=actual_id).first()
            if test_doctor:
                logger.info("✅ SUCCESS: Found doctor by ID!")
                return actual_id
            else:
                logger.error("❌ FAILED: Could not find doctor by ID")
                
                # Test with explicit string conversion
                logger.info("3. Trying with explicit string conversion...")
                test_doctor2 = Doctor.objects(doctor_ID=str(actual_id)).first()
                if test_doctor2:
                    logger.info("✅ Found with str() conversion")
                else:
                    logger.error("❌ Still failed with str() conversion")
                
                # Test with stripped string
                logger.info("4. Trying with stripped string...")
                test_doctor3 = Doctor.objects(doctor_ID=actual_id.strip()).first()
                if test_doctor3:
                    logger.info("✅ Found with strip()")
                else:
                    logger.error("❌ Still failed with strip()")
                
                return None
        else:
            logger.error("❌ Could not find doctor by email")
            return None
            
    except Exception as e:
        logger.error(f"Error in simple test: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_get_doctor_function():
    """Test the actual get_doctor_by_id function"""
    logger.info("\n🔍 Testing get_doctor_by_id function...")
    
    try:
        from repository.doctor_repository import get_doctor_by_id
        doctor = get_doctor_by_id("D15")
        logger.info("✅ get_doctor_by_id worked!")
        return True
    except Exception as e:
        logger.error(f"❌ get_doctor_by_id failed: {e}")
        return False

if __name__ == "__main__":
    actual_id = simple_doctor_test()
    if actual_id:
        logger.info(f"\n🎯 Actual doctor ID found: '{actual_id}'")
        
        # Test the repository function
        test_get_doctor_function()
    else:
        logger.error("Could not determine actual doctor ID")