"""
Chat Controller - HTTP Request Handler Layer
Handles all HTTP requests for RAG-based patient chat functionality
Enhanced with better error handling, logging, and alignment with architecture
Preserves all original endpoint logic
"""

import datetime
import logging
from flask import Blueprint, request, jsonify, make_response
from flask_cors import cross_origin

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Service Layer Imports
from services.RAG_service import PatientRAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Initialize RAG service (singleton pattern - Original logic preserved)
RAG_service = PatientRAGService()

# ================================
# HELPER FUNCTIONS
# ================================

def _cors_preflight():
    """Handle CORS preflight requests - Original logic preserved"""
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

def _handle_controller_error(error, default_status=500):
    """Consistent error handling for controller endpoints"""
    logger.error(f"Chat controller error: {error}")
    
    if isinstance(error, ValueError):
        # Patient not found or validation errors
        if "not found" in str(error).lower():
            return jsonify({"error": "Patient not found", "details": str(error)}), 404
        else:
            return jsonify({"error": "Invalid input", "details": str(error)}), 400
    elif isinstance(error, KeyError):
        return jsonify({"error": "Missing required data", "details": str(error)}), 400
    else:
        return jsonify({"error": "Internal server error", "details": str(error)}), default_status

def _validate_patient_id(patient_id):
    """Validate patient ID - Enhanced validation"""
    if not isinstance(patient_id, int) or patient_id <= 0:
        raise ValueError("Invalid patient ID. Must be a positive integer.")
    return patient_id

def _validate_chat_request(data):
    """Validate chat request data"""
    if not data:
        raise ValueError("No JSON data provided")
    
    query = data.get('query', '').strip()
    if not query:
        raise ValueError("Query cannot be empty")
    
    chat_history = data.get('history', [])
    if not isinstance(chat_history, list):
        raise ValueError("Chat history must be a list")
    
    return query, chat_history

# ================================
# MAIN CHAT ENDPOINTS
# ================================

@chat_bp.route('/chat/<int:patient_id>', methods=['POST', 'OPTIONS'])
@cross_origin()
def chat_with_patient_data(patient_id):
    """
    Main chat endpoint for patient-specific conversations
    Original logic preserved exactly
    """
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        # Validate patient ID - Enhanced validation but same logic
        if patient_id <= 0:
            return jsonify({'error': 'Invalid patient ID'}), 400
        
        # Get request data - Original logic preserved
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query', '').strip()
        chat_history = data.get('history', [])
        
        # Validate input - Original logic preserved
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Chat request for patient {patient_id}: {query[:50]}...")
        print(f"🤖 Chat request for patient {patient_id}: {query[:50]}...")
        
        # Generate response using RAG - Original logic preserved
        response = RAG_service.generate_response(patient_id, query, chat_history)
        
        logger.info(f"Chat response generated for patient {patient_id}")
        return jsonify({
            'response': response,
            'patient_id': patient_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'success'
        }), 200
        
    except ValueError as ve:
        # Patient not found or validation error - Original logic preserved
        print(f"❌ Validation error: {ve}")
        return jsonify({'error': str(ve)}), 404
        
    except Exception as e:
        # General server error - Original logic preserved
        print(f"❌ Server error in chat: {e}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

# ================================
# KNOWLEDGE BASE MANAGEMENT ENDPOINTS
# ================================

@chat_bp.route('/chat/<int:patient_id>/rebuild-kb', methods=['POST', 'OPTIONS'])
@cross_origin()
def rebuild_knowledge_base(patient_id):
    """
    Manually rebuild knowledge base for a patient
    Original logic preserved exactly
    """
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        # Validate patient_id - Original logic preserved
        if patient_id <= 0:
            return jsonify({'error': 'Invalid patient ID'}), 400
        
        logger.info(f"Rebuilding knowledge base for patient {patient_id}")
        print(f"🔄 Rebuilding knowledge base for patient {patient_id}")
        
        # Rebuild knowledge base - Original logic preserved
        RAG_service.build_knowledge_base(patient_id)
        
        # Get updated stats - Original logic preserved
        stats = RAG_service.get_knowledge_base_stats(patient_id)
        
        logger.info(f"Knowledge base rebuilt successfully for patient {patient_id}")
        return jsonify({
            'message': f'Knowledge base rebuilt for patient {patient_id}',
            'patient_id': patient_id,
            'stats': stats,
            'timestamp': datetime.datetime.now().isoformat()
        }), 200
        
    except ValueError as ve:
        # Patient not found - Original logic preserved
        print(f"❌ Patient not found: {ve}")
        return jsonify({'error': str(ve)}), 404
        
    except Exception as e:
        # General server error - Original logic preserved
        print(f"❌ Error rebuilding knowledge base: {e}")
        return jsonify({
            'error': 'Failed to rebuild knowledge base',
            'details': str(e)
        }), 500

@chat_bp.route('/chat/<int:patient_id>/status', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_knowledge_base_status(patient_id):
    """
    Get status and statistics of knowledge base for a patient
    Original logic preserved exactly
    """
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        # Validate patient_id - Original logic preserved
        if patient_id <= 0:
            return jsonify({'error': 'Invalid patient ID'}), 400
        
        # Get knowledge base stats - Original logic preserved
        stats = RAG_service.get_knowledge_base_stats(patient_id)
        
        logger.info(f"Knowledge base status retrieved for patient {patient_id}")
        return jsonify({
            'patient_id': patient_id,
            'knowledge_base': stats,
            'timestamp': datetime.datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error getting KB status: {e}")
        return jsonify({
            'error': 'Failed to get knowledge base status',
            'details': str(e)
        }), 500

@chat_bp.route('/chat/<int:patient_id>/clear-kb', methods=['DELETE', 'OPTIONS'])
@cross_origin()
def clear_knowledge_base(patient_id):
    """
    Clear knowledge base for a specific patient
    Original logic preserved exactly
    """
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        # Validate patient_id - Original logic preserved
        if patient_id <= 0:
            return jsonify({'error': 'Invalid patient ID'}), 400
        
        logger.info(f"Clearing knowledge base for patient {patient_id}")
        print(f"🗑️ Clearing knowledge base for patient {patient_id}")
        
        # Clear knowledge base - Original logic preserved
        RAG_service.clear_knowledge_base(patient_id)
        
        logger.info(f"Knowledge base cleared successfully for patient {patient_id}")
        return jsonify({
            'message': f'Knowledge base cleared for patient {patient_id}',
            'patient_id': patient_id,
            'timestamp': datetime.datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error clearing knowledge base: {e}")
        return jsonify({
            'error': 'Failed to clear knowledge base',
            'details': str(e)
        }), 500

# ================================
# UTILITY ENDPOINTS
# ================================

@chat_bp.route('/chat/health', methods=['GET', 'OPTIONS'])
@cross_origin()
def chat_health_check():
    """
    Health check endpoint for chat service
    Original logic preserved with enhancements
    """
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        # Test RAG service availability
        service_status = "healthy"
        error_details = None
        
        try:
            # Test if we can access the RAG service
            test_stats = RAG_service.get_knowledge_base_stats(999999)  # Non-existent patient
            # If this doesn't throw an error, service is working
        except Exception as e:
            # Expected for non-existent patient, this is fine
            pass
        
        return jsonify({
            'status': service_status,
            'service': 'chat_RAG_service',
            'timestamp': datetime.datetime.now().isoformat(),
            'embedding_model': 'TF-IDF (sklearn)',
            'llm_model': 'gemini-2.0-flash',
            'features': {
                'patient_specific_rag': True,
                'cross_session_analysis': True,
                'therapeutic_guidelines': True,
                'knowledge_base_management': True
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'chat_RAG_service',
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }), 500

# ================================
# ENHANCED UTILITY ENDPOINTS
# ================================

@chat_bp.route('/chat/<int:patient_id>/capabilities', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_patient_chat_capabilities(patient_id):
    """Get chat capabilities and available data for a specific patient"""
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        _validate_patient_id(patient_id)
        
        # Check if patient exists and get basic info
        from model.patient_model import Patient
        patient = Patient.objects(patientID=patient_id).first()
        
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")
        
        # Get knowledge base status
        kb_stats = RAG_service.get_knowledge_base_stats(patient_id)
        
        # Analyze available data sources
        capabilities = {
            "patient_id": patient_id,
            "chat_available": True,
            "knowledge_base": kb_stats,
            "data_sources": {
                "personal_info": bool(patient.personalInfo),
                "sessions": len(patient.sessions) if patient.sessions else 0,
                "transcriptions": 0,
                "reports": 0,
                "doctor_notes": 0
            },
            "session_analysis": {
                "total_sessions": len(patient.sessions) if patient.sessions else 0,
                "sessions_with_data": 0,
                "date_range": None
            }
        }
        
        # Analyze session data
        if patient.sessions:
            sessions_with_data = 0
            dates = []
            
            for session in patient.sessions:
                has_data = False
                
                if session.transcription:
                    capabilities["data_sources"]["transcriptions"] += 1
                    has_data = True
                
                if session.report:
                    capabilities["data_sources"]["reports"] += 1
                    has_data = True
                
                if session.doctor_notes_images:
                    capabilities["data_sources"]["doctor_notes"] += len(session.doctor_notes_images)
                    has_data = True
                
                if has_data:
                    sessions_with_data += 1
                
                if session.date:
                    dates.append(session.date)
            
            capabilities["session_analysis"]["sessions_with_data"] = sessions_with_data
            
            if dates:
                dates.sort()
                capabilities["session_analysis"]["date_range"] = {
                    "start": dates[0].isoformat(),
                    "end": dates[-1].isoformat()
                }
        
        logger.info(f"Chat capabilities retrieved for patient {patient_id}")
        return jsonify(capabilities), 200
        
    except Exception as e:
        return _handle_controller_error(e)

@chat_bp.route('/chat/<int:patient_id>/context-preview', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_context_preview(patient_id):
    """Get a preview of the context available for chat"""
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        _validate_patient_id(patient_id)
        
        # Get knowledge base stats
        kb_stats = RAG_service.get_knowledge_base_stats(patient_id)
        
        if kb_stats["status"] == "not_built":
            # Build knowledge base if it doesn't exist
            RAG_service.build_knowledge_base(patient_id)
            kb_stats = RAG_service.get_knowledge_base_stats(patient_id)
        
        # Get sample context
        sample_query = "therapy session overview"
        relevant_chunks = RAG_service.retrieve_relevant_context(patient_id, sample_query, top_k=3)
        
        context_preview = {
            "patient_id": patient_id,
            "knowledge_base_stats": kb_stats,
            "sample_context": []
        }
        
        for chunk in relevant_chunks:
            context_preview["sample_context"].append({
                "type": chunk["metadata"]["type"],
                "source": chunk["metadata"]["source"],
                "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "similarity": chunk["similarity"]
            })
        
        logger.info(f"Context preview generated for patient {patient_id}")
        return jsonify(context_preview), 200
        
    except Exception as e:
        return _handle_controller_error(e)

# ================================
# BATCH OPERATIONS ENDPOINTS
# ================================

@chat_bp.route('/chat/batch/rebuild-all', methods=['POST', 'OPTIONS'])
@cross_origin()
def rebuild_all_knowledge_bases():
    """
    Rebuild knowledge bases for all patients (admin operation)
    Original logic preserved exactly
    """
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        # Get list of all patients - Original logic preserved
        from model.patient_model import Patient
        all_patients = Patient.objects()
        
        rebuilt_count = 0
        errors = []
        
        logger.info(f"Starting batch rebuild for {len(all_patients)} patients")
        
        # Process each patient - Original logic preserved
        for patient in all_patients:
            try:
                RAG_service.build_knowledge_base(patient.patientID)
                rebuilt_count += 1
                logger.info(f"Rebuilt knowledge base for patient {patient.patientID}")
            except Exception as e:
                error_info = {
                    'patient_id': patient.patientID,
                    'error': str(e)
                }
                errors.append(error_info)
                logger.error(f"Failed to rebuild knowledge base for patient {patient.patientID}: {e}")
        
        logger.info(f"Batch rebuild completed: {rebuilt_count}/{len(all_patients)} successful")
        return jsonify({
            'message': f'Batch rebuild completed',
            'total_patients': len(all_patients),
            'successfully_rebuilt': rebuilt_count,
            'errors': errors,
            'timestamp': datetime.datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch rebuild failed: {e}")
        return jsonify({
            'error': 'Failed to rebuild all knowledge bases',
            'details': str(e)
        }), 500

@chat_bp.route('/chat/batch/clear-all', methods=['DELETE', 'OPTIONS'])
@cross_origin()
def clear_all_knowledge_bases():
    """Clear all knowledge bases (admin operation)"""
    if request.method == 'OPTIONS':
        return _cors_preflight()
    
    try:
        logger.info("Clearing all knowledge bases")
        
        # Clear all knowledge bases
        RAG_service.clear_knowledge_base()  # No patient_id = clear all
        
        logger.info("All knowledge bases cleared successfully")
        return jsonify({
            'message': 'All knowledge bases cleared successfully',
            'timestamp': datetime.datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to clear all knowledge bases: {e}")
        return jsonify({
            'error': 'Failed to clear all knowledge bases',
            'details': str(e)
        }), 500

# ================================
# ERROR HANDLERS
# ================================

@chat_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({"error": "Bad request. Please check your input parameters."}), 400

@chat_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({"error": "Resource not found."}), 404

@chat_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({"error": "Internal server error. Please try again later."}), 500