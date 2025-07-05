# from flask import Blueprint, request, jsonify, Response, send_file
# from flask_cors import cross_origin
# from werkzeug.utils import secure_filename
# import os
# import tempfile
# import traceback
# from io import BytesIO
# from bson import ObjectId

# from services.speech_transcription_service import speech_recognition_and_transcription
# from repository.patient_repository import save_speech_video_to_session, get_patient_by_id
# from config import fs

# # Initialize the blueprint
# speech_recognition_bp = Blueprint('speech_recognition', __name__)

# # ================================
# # CONSTANTS AND CONFIGURATION
# # ================================

# ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.flac'}
# ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv'}
# ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB limit


# # ================================
# # UTILITY FUNCTIONS
# # ================================

# def _cors_preflight():
#     """Handle CORS preflight requests"""
#     resp = jsonify({})
#     resp.headers["Access-Control-Allow-Origin"] = "*"
#     resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
#     resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
#     return resp


# def _validate_file(file, max_size=MAX_FILE_SIZE):
#     """Validate uploaded file type and size"""
#     if not file or file.filename == '':
#         raise ValueError("No file selected")
    
#     filename = secure_filename(file.filename)
#     file_ext = os.path.splitext(filename)[1].lower()
    
#     if file_ext not in ALLOWED_EXTENSIONS:
#         raise ValueError(
#             f"Unsupported file type: {file_ext}. "
#             f"Supported types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
#         )
    
#     # Check file size if possible
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)  # Reset to beginning
    
#     if file_size > max_size:
#         raise ValueError(f"File too large. Maximum size: {max_size // (1024*1024)}MB")
    
#     return filename, file_ext


# def _get_session_by_id(patient_id, session_id):
#     """Get a specific session for a patient"""
#     patient = get_patient_by_id(patient_id)
    
#     for session in patient.sessions:
#         if session.session_id == session_id:
#             return patient, session
    
#     raise ValueError("Session not found")


# def _handle_error(error, default_status=500):
#     """Centralized error handling"""
#     if isinstance(error, ValueError):
#         status = 400
#     elif isinstance(error, FileNotFoundError):
#         status = 404
#     else:
#         status = default_status
#         traceback.print_exc()
    
#     return jsonify({
#         "error": str(error),
#         "type": type(error).__name__
#     }), status


# # ================================
# # UPLOAD ENDPOINTS
# # ================================

# @speech_recognition_bp.route(
#     '/patients/<int:patient_id>/sessions/<int:session_id>/upload-transcription-video',
#     methods=['POST', 'OPTIONS']
# )
# @cross_origin()
# def upload_transcription_video(patient_id, session_id):
#     """Upload audio/video file for speech recognition and transcription"""
#     if request.method == 'OPTIONS':
#         return _cors_preflight()
    
#     try:
#         # Validate request
#         if 'video' not in request.files:
#             raise ValueError("No video file uploaded")
        
#         video_file = request.files['video']
#         filename, file_ext = _validate_file(video_file)
        
#         print(f"📤 Uploading {file_ext} file: {filename} for patient {patient_id}, session {session_id}")
        
#         # Save file temporarily and upload to GridFS
#         with tempfile.TemporaryDirectory() as tmpdir:
#             video_path = os.path.join(tmpdir, filename)
#             video_file.save(video_path)
            
#             # Save to database
#             video_file_id = save_speech_video_to_session(
#                 patient_id, session_id, video_path, filename
#             )
            
#             print(f"✅ File uploaded successfully with ID: {video_file_id}")
            
#             return jsonify({
#                 "message": "File uploaded successfully for transcription",
#                 "video_file_id": str(video_file_id),
#                 "filename": filename,
#                 "file_type": file_ext,
#                 "patient_id": patient_id,
#                 "session_id": session_id
#             }), 200
            
#     except Exception as e:
#         return _handle_error(e)


# # ================================
# # ANALYSIS ENDPOINTS
# # ================================

# @speech_recognition_bp.route(
#     '/transcription/analyze/<string:file_id>/<int:patient_id>/<int:session_id>',
#     methods=['GET', 'OPTIONS']
# )
# @cross_origin()
# def analyze_and_transcribe(file_id, patient_id, session_id):
#     """Analyze speech and generate transcription for a given file ID"""
#     if request.method == 'OPTIONS':
#         return _cors_preflight()
    
#     try:
#         print(f"🎤 Starting transcription analysis...")
#         print(f"   File ID: {file_id}")
#         print(f"   Patient: {patient_id}, Session: {session_id}")
        
#         # Process speech recognition and transcription
#         pdf_file_id = speech_recognition_and_transcription(file_id, patient_id, session_id)
        
#         print(f"✅ Transcription completed successfully")
#         print(f"   Generated PDF ID: {pdf_file_id}")
        
#         return jsonify({
#             "message": "Speech recognition and transcription completed successfully",
#             "pdf_file_id": str(pdf_file_id),
#             "patient_id": patient_id,
#             "session_id": session_id,
#             "download_url": f"/transcription/download/{pdf_file_id}",
#             "view_url": f"/transcription/view/{pdf_file_id}"
#         }), 200
        
#     except Exception as e:
#         return _handle_error(e)


# # ================================
# # DOWNLOAD/VIEW ENDPOINTS
# # ================================

# @speech_recognition_bp.route(
#     '/transcription/download/<string:pdf_file_id>',
#     methods=['GET', 'OPTIONS']
# )
# @cross_origin()
# def download_transcription_pdf(pdf_file_id):
#     """Download the transcription PDF by file ID"""
#     if request.method == 'OPTIONS':
#         return _cors_preflight()
    
#     try:
#         print(f"📥 Downloading transcription PDF: {pdf_file_id}")
        
#         # Get PDF from GridFS
#         grid_out = fs.get(ObjectId(pdf_file_id))
#         pdf_data = grid_out.read()
#         filename = grid_out.filename or f"transcription_{pdf_file_id}.pdf"
        
#         return send_file(
#             BytesIO(pdf_data),
#             mimetype='application/pdf',
#             as_attachment=True,
#             download_name=filename
#         )
        
#     except Exception as e:
#         print(f"❌ Error downloading PDF: {str(e)}")
#         return _handle_error(e, 404)


# @speech_recognition_bp.route(
#     '/transcription/view/<string:pdf_file_id>',
#     methods=['GET', 'OPTIONS']
# )
# @cross_origin()
# def view_transcription_pdf(pdf_file_id):
#     """View the transcription PDF in browser by file ID"""
#     if request.method == 'OPTIONS':
#         return _cors_preflight()
    
#     try:
#         print(f"👁️ Viewing transcription PDF: {pdf_file_id}")
        
#         # Get PDF from GridFS
#         grid_out = fs.get(ObjectId(pdf_file_id))
#         pdf_data = grid_out.read()
        
#         return Response(
#             pdf_data,
#             mimetype='application/pdf',
#             headers={
#                 'Content-Disposition': 'inline',
#                 'Access-Control-Allow-Origin': '*'
#             }
#         )
        
#     except Exception as e:
#         print(f"❌ Error viewing PDF: {str(e)}")
#         return _handle_error(e, 404)


# # ================================
# # STATUS ENDPOINTS
# # ================================

# @speech_recognition_bp.route(
#     '/patients/<int:patient_id>/sessions/<int:session_id>/transcription-status',
#     methods=['GET', 'OPTIONS']
# )
# @cross_origin()
# def get_transcription_status(patient_id, session_id):
#     """Check if transcription exists for a specific patient session"""
#     if request.method == 'OPTIONS':
#         return _cors_preflight()
    
#     try:
#         patient, session = _get_session_by_id(patient_id, session_id)
        
#         # Check transcription status
#         has_transcription = session.report is not None
#         transcription_file_id = str(session.report) if session.report else None
        
#         # Check if session has uploaded media for transcription
#         has_media = False
#         media_type = None
        
#         # Check various media sources
#         if session.audio_files:
#             has_media = True
#             media_type = "audio"
#         elif session.video_files:
#             has_media = True
#             media_type = "video"
#         elif session.feature_data:
#             speech_data = session.feature_data.get('Speech', {})
#             if speech_data.get('video_files'):
#                 has_media = True
#                 media_type = "speech_video"
        
#         status = {
#             "patient_id": patient_id,
#             "session_id": session_id,
#             "has_transcription": has_transcription,
#             "has_media": has_media,
#             "media_type": media_type,
#             "transcription_file_id": transcription_file_id,
#             "can_generate_transcription": has_media and not has_transcription,
#             "needs_media_upload": not has_media
#         }
        
#         # Add URLs if transcription exists
#         if has_transcription:
#             status.update({
#                 "download_url": f"/transcription/download/{transcription_file_id}",
#                 "view_url": f"/transcription/view/{transcription_file_id}"
#             })
        
#         return jsonify(status), 200
        
#     except Exception as e:
#         return _handle_error(e)


# @speech_recognition_bp.route(
#     '/patients/<int:patient_id>/transcriptions/summary',
#     methods=['GET', 'OPTIONS']
# )
# @cross_origin()
# def get_patient_transcriptions_summary(patient_id):
#     """Get a summary of all transcriptions for a patient"""
#     if request.method == 'OPTIONS':
#         return _cors_preflight()
    
#     try:
#         patient = get_patient_by_id(patient_id)
        
#         transcriptions = []
#         total_transcriptions = 0
        
#         for session in patient.sessions:
#             if session.report:
#                 session_info = {
#                     "session_id": session.session_id,
#                     "session_date": session.date.isoformat() if session.date else None,
#                     "session_type": session.session_type,
#                     "transcription_file_id": str(session.report),
#                     "download_url": f"/transcription/download/{session.report}",
#                     "view_url": f"/transcription/view/{session.report}"
#                 }
#                 transcriptions.append(session_info)
#                 total_transcriptions += 1
        
#         summary = {
#             "patient_id": patient_id,
#             "patient_name": getattr(patient.personalInfo, 'full_name', 'Unknown'),
#             "total_transcriptions": total_transcriptions,
#             "total_sessions": len(patient.sessions),
#             "transcription_coverage": f"{(total_transcriptions/len(patient.sessions)*100):.1f}%" if patient.sessions else "0%",
#             "transcriptions": transcriptions
#         }
        
#         return jsonify(summary), 200
        
#     except Exception as e:
#         return _handle_error(e)


# # ================================
# # UTILITY ENDPOINTS
# # ================================

# @speech_recognition_bp.route('/transcription/health', methods=['GET'])
# def health_check():
#     """Health check endpoint for transcription service"""
#     try:
#         # Test GridFS connection
#         fs.list()
        
#         return jsonify({
#             "status": "healthy",
#             "service": "speech_recognition",
#             "features": {
#                 "supported_audio_formats": list(ALLOWED_AUDIO_EXTENSIONS),
#                 "supported_video_formats": list(ALLOWED_VIDEO_EXTENSIONS),
#                 "max_file_size_mb": MAX_FILE_SIZE // (1024*1024)
#             }
#         }), 200
        
#     except Exception as e:
#         return jsonify({
#             "status": "unhealthy",
#             "error": str(e)
#         }), 503


# @speech_recognition_bp.route('/transcription/formats', methods=['GET'])
# def get_supported_formats():
#     """Get list of supported file formats"""
#     return jsonify({
#         "audio_formats": sorted(list(ALLOWED_AUDIO_EXTENSIONS)),
#         "video_formats": sorted(list(ALLOWED_VIDEO_EXTENSIONS)),
#         "all_formats": sorted(list(ALLOWED_EXTENSIONS)),
#         "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
#         "recommendations": {
#             "audio": "MP3 or WAV for best quality",
#             "video": "MP4 for best compatibility",
#             "max_duration": "Recommended under 60 minutes for faster processing"
#         }
#     }), 200

from flask import Blueprint, request, jsonify, Response, send_file
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import os
import tempfile
import traceback
from io import BytesIO
from bson import ObjectId

from services.speech_transcription_service import speech_recognition_and_transcription
from repository.patient_repository import (
    save_speech_video_to_session,
    get_patient_by_id,
    update_transcription_pdf_reference,
)
from config import fs

speech_recognition_bp = Blueprint('speech_recognition', __name__)

ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.flac'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

def _cors_preflight():
    resp = jsonify({})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

def _validate_file(file, max_size=MAX_FILE_SIZE):
    if not file or file.filename == '':
        raise ValueError("No file selected")
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > max_size:
        raise ValueError(f"File too large (> {max_size//(1024*1024)}MB)")
    return filename, ext

def _get_session(patient_id, session_id):
    patient = get_patient_by_id(patient_id)
    for sess in patient.sessions:
        if sess.session_id == session_id:
            return patient, sess
    raise ValueError("Session not found")

def _handle_error(err, default_status=500):
    traceback.print_exc()
    code = 400 if isinstance(err, ValueError) else default_status
    return jsonify(error=str(err), type=type(err).__name__), code

# ─── UPLOAD ─────────────────────────────────────────────────────────────

@speech_recognition_bp.route(
    '/patients/<int:patient_id>/sessions/<int:session_id>/upload-transcription-video',
    methods=['POST','OPTIONS']
)
@cross_origin()
def upload_transcription_video(patient_id, session_id):
    if request.method == 'OPTIONS':
        return _cors_preflight()
    try:
        if 'video' not in request.files:
            raise ValueError("No video file uploaded")
        vid = request.files['video']
        filename, ext = _validate_file(vid)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, filename)
            vid.save(path)
            file_id = save_speech_video_to_session(patient_id, session_id, path, filename)
        return jsonify(
            message="File uploaded",
            video_file_id=str(file_id),
            patient_id=patient_id,
            session_id=session_id
        ), 200
    except Exception as e:
        return _handle_error(e)

# ─── ANALYZE & TRANSCRIBE ───────────────────────────────────────────────

@speech_recognition_bp.route(
    '/transcription/analyze/<string:file_id>/<int:patient_id>/<int:session_id>',
    methods=['GET','OPTIONS']
)
@cross_origin()
def analyze_and_transcribe(file_id, patient_id, session_id):
    if request.method == 'OPTIONS':
        return _cors_preflight()
    try:
        pdf_id = speech_recognition_and_transcription(file_id, patient_id, session_id)
        # store under session.transcription
        update_transcription_pdf_reference(patient_id, session_id, pdf_id)
        return jsonify(
            message="Transcription complete",
            transcription_file_id=str(pdf_id),
            download_url=f"/transcription/download/{pdf_id}",
            view_url=f"/transcription/view/{pdf_id}"
        ), 200
    except Exception as e:
        return _handle_error(e)

# ─── DOWNLOAD ───────────────────────────────────────────────────────────

@speech_recognition_bp.route(
    '/transcription/download/<string:transcription_id>',
    methods=['GET','OPTIONS']
)
@cross_origin()
def download_transcription_pdf(transcription_id):
    if request.method == 'OPTIONS':
        return _cors_preflight()
    try:
        grid_out = fs.get(ObjectId(transcription_id))
        data = grid_out.read()
        fname = grid_out.filename or f"transcription_{transcription_id}.pdf"
        return send_file(
            BytesIO(data),
            mimetype='application/pdf',
            as_attachment=True,
            attachment_filename=fname
        )
    except Exception as e:
        return _handle_error(e, 404)

# ─── VIEW ───────────────────────────────────────────────────────────────

@speech_recognition_bp.route(
    '/transcription/view/<string:transcription_id>',
    methods=['GET','OPTIONS']
)
@cross_origin()
def view_transcription_pdf(transcription_id):
    if request.method == 'OPTIONS':
        return _cors_preflight()
    try:
        grid_out = fs.get(ObjectId(transcription_id))
        data = grid_out.read()
        return Response(
            data,
            mimetype='application/pdf',
            headers={'Content-Disposition':'inline','Access-Control-Allow-Origin':'*'}
        )
    except Exception as e:
        return _handle_error(e, 404)

# ─── STATUS ─────────────────────────────────────────────────────────────

@speech_recognition_bp.route(
    '/patients/<int:patient_id>/sessions/<int:session_id>/transcription-status',
    methods=['GET','OPTIONS']
)
@cross_origin()
def get_transcription_status(patient_id, session_id):
    if request.method == 'OPTIONS':
        return _cors_preflight()
    try:
        _, sess = _get_session(patient_id, session_id)
        _id = getattr(sess, 'transcription', None)
        has = _id is not None
        payload = {
            "patient_id": patient_id,
            "session_id": session_id,
            "has_transcription": has,
            "transcription_file_id": str(_id) if has else None,
            "can_generate_transcription": hasattr(sess, 'video_files') or hasattr(sess, 'audio_files'),
            "needs_media_upload": not (hasattr(sess, 'video_files') or hasattr(sess, 'audio_files'))
        }
        if has:
            payload["download_url"] = f"/transcription/download/{_id}"
            payload["view_url"] = f"/transcription/view/{_id}"
        return jsonify(payload), 200
    except Exception as e:
        return _handle_error(e)

# ─── SUMMARY ────────────────────────────────────────────────────────────

@speech_recognition_bp.route(
    '/patients/<int:patient_id>/transcriptions/summary',
    methods=['GET','OPTIONS']
)
@cross_origin()
def get_patient_transcriptions_summary(patient_id):
    if request.method == 'OPTIONS':
        return _cors_preflight()
    try:
        patient = get_patient_by_id(patient_id)
        lst = []
        for sess in patient.sessions:
            if getattr(sess, 'transcription', None):
                tid = str(sess.transcription)
                lst.append({
                    "session_id": sess.session_id,
                    "session_date": sess.date.isoformat() if sess.date else None,
                    "transcription_file_id": tid,
                    "download_url": f"/transcription/download/{tid}",
                    "view_url":    f"/transcription/view/{tid}"
                })
        total = len(patient.sessions)
        done  = len(lst)
        return jsonify({
            "patient_id": patient_id,
            "patient_name": patient.personalInfo.fullName,
            "total_sessions": total,
            "total_transcriptions": done,
            "transcription_coverage": f"{(done/total*100):.1f}%" if total else "0%",
            "transcriptions": lst
        }), 200
    except Exception as e:
        return _handle_error(e)

# ─── HEALTH & FORMATS ───────────────────────────────────────────────────

@speech_recognition_bp.route('/transcription/health', methods=['GET'])
def health_check():
    try:
        fs.list()
        return jsonify(status="healthy"), 200
    except Exception as e:
        return jsonify(status="unhealthy", error=str(e)), 503

@speech_recognition_bp.route('/transcription/formats', methods=['GET'])
def get_supported_formats():
    return jsonify(
        audio_formats=sorted(list(ALLOWED_AUDIO_EXTENSIONS)),
        video_formats=sorted(list(ALLOWED_VIDEO_EXTENSIONS)),
        max_file_size_mb=MAX_FILE_SIZE//(1024*1024)
    ), 200
