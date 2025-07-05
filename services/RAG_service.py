# services/rag_service.py
import os
import io
import tempfile
import datetime
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import gridfs
from pymongo import MongoClient
import numpy as np
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from bson import ObjectId

# Initialize components
api_key = "AIzaSyBsDcl5tRJd6FR0fy0pNvwv76-S5QrVvK4"
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize TF-IDF vectorizer instead of sentence transformers
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)

mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["seranityAI"]
fs = gridfs.GridFS(db)

class PatientRAGService:
    def __init__(self):
        self.knowledge_base = {}  # patient_id -> {chunks, embeddings, metadata, vectorizer}
        
    def extract_text_from_pdf(self, pdf_file_id: str) -> str:
        """Extract text from PDF stored in GridFS"""
        try:
            grid_out = fs.get(ObjectId(pdf_file_id))
            pdf_data = grid_out.read()
            
            with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_data)
                tmp_file.flush()
                
                with open(tmp_file.name, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def _extract_session_summaries(self, patient) -> str:
        """Extract comprehensive session summaries for cross-session analysis"""
        if not patient.sessions:
            return "No therapy sessions have been conducted yet."
        
        session_summaries = []
        session_summaries.append(f"THERAPY SESSION OVERVIEW FOR {patient.personalInfo.full_name}")
        session_summaries.append(f"Total Sessions Conducted: {len(patient.sessions)}")
        session_summaries.append("=" * 50)
        
        # Sort sessions by date for chronological analysis
        sorted_sessions = sorted(patient.sessions, 
                               key=lambda x: x.date if x.date else datetime.datetime.min)
        
        for i, session in enumerate(sorted_sessions, 1):
            session_summary = [
                f"\nSESSION {session.session_id} SUMMARY:",
                f"Chronological Order: Session {i} of {len(patient.sessions)}",
                f"Date: {session.date.strftime('%Y-%m-%d') if session.date else 'Not recorded'}",
                f"Session Type: {session.session_type or 'Not specified'}",
                f"Feature Analysis: {session.feature_type or 'Not specified'}",
                f"Duration: {session.duration or 'Not recorded'}",
            ]
            
            # Add session-specific insights
            if session.doctor_notes:
                session_summary.append(f"Doctor's Notes: {session.doctor_notes}")
            
            if session.text:
                # Truncate long transcriptions
                text_preview = session.text[:200] + "..." if len(session.text) > 200 else session.text
                session_summary.append(f"Session Content Preview: {text_preview}")
            
            # Add analysis status
            analysis_status = []
            if session.report:
                analysis_status.append("✓ PDF Report Available")
            if session.audio_files:
                analysis_status.append("✓ Audio Analysis")
            if session.video_files:
                analysis_status.append("✓ Video Analysis")
            if session.feature_data:
                if 'FER' in str(session.feature_data):
                    analysis_status.append("✓ Facial Emotion Recognition")
                if 'Speech' in str(session.feature_data):
                    analysis_status.append("✓ Speech Tone Analysis")
            
            if analysis_status:
                session_summary.append(f"Available Analysis: {', '.join(analysis_status)}")
            else:
                session_summary.append("Available Analysis: Basic session record only")
            
            session_summaries.extend(session_summary)
            session_summaries.append("-" * 30)
        
        # Add progression analysis prompt
        session_summaries.extend([
            "\nCROSS-SESSION ANALYSIS GUIDANCE:",
            "- Compare emotional patterns across sessions",
            "- Track therapeutic progress over time", 
            "- Identify improvement trends or concerning patterns",
            "- Note changes in patient engagement and response",
            "- Assess effectiveness of therapeutic interventions",
            f"- Sessions span from {sorted_sessions[0].date.strftime('%Y-%m-%d') if sorted_sessions[0].date else 'Unknown'} to {sorted_sessions[-1].date.strftime('%Y-%m-%d') if sorted_sessions[-1].date else 'Unknown'}",
        ])
        
        return "\n".join(session_summaries)
    
    def build_knowledge_base(self, patient_id: int):
        """Build knowledge base for a specific patient with enhanced cross-session analysis"""
        try:
            # Get patient data
            from model.patient_model import Patient
            patient = Patient.objects(patientID=patient_id).first()
            
            if not patient:
                raise ValueError(f"Patient {patient_id} not found")
            
            all_chunks = []
            all_metadata = []
            
            # 1. Extract patient personal information
            personal_info = self._extract_patient_info(patient)
            personal_chunks = self.chunk_text(personal_info)
            
            for chunk in personal_chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'type': 'patient_info',
                    'source': 'personal_data',
                    'patient_id': patient_id
                })
            
            # 2. Extract session summaries for cross-session analysis
            session_summaries = self._extract_session_summaries(patient)
            summary_chunks = self.chunk_text(session_summaries)
            
            for chunk in summary_chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'type': 'session_summaries',
                    'source': 'cross_session_analysis',
                    'patient_id': patient_id,
                    'total_sessions': len(patient.sessions)
                })
            
            # 3. Extract individual session reports (PDFs)
            for session in patient.sessions:
                if session.report:  # If session has a PDF report
                    pdf_text = self.extract_text_from_pdf(str(session.report))
                    if pdf_text:
                        session_chunks = self.chunk_text(pdf_text)
                        
                        for chunk in session_chunks:
                            all_chunks.append(chunk)
                            all_metadata.append({
                                'type': 'session_report',
                                'source': f'session_{session.session_id}',
                                'session_id': session.session_id,
                                'session_date': session.date.isoformat() if session.date else None,
                                'patient_id': patient_id
                            })
            
            # 4. Add general therapy guidelines
            therapy_guidelines = self._get_therapy_guidelines()
            guideline_chunks = self.chunk_text(therapy_guidelines)
            
            for chunk in guideline_chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'type': 'therapy_guidelines',
                    'source': 'clinical_guidelines',
                    'patient_id': patient_id
                })
            
            # Generate embeddings using TF-IDF
            if all_chunks:
                # Create a fresh vectorizer for this patient
                patient_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                embeddings = patient_vectorizer.fit_transform(all_chunks)
                
                # Store in knowledge base
                self.knowledge_base[patient_id] = {
                    'chunks': all_chunks,
                    'embeddings': embeddings,
                    'metadata': all_metadata,
                    'vectorizer': patient_vectorizer
                }
                
                print(f"✅ Knowledge base built for patient {patient_id}: {len(all_chunks)} chunks")
                print(f"   - Personal info chunks: {len([m for m in all_metadata if m['type'] == 'patient_info'])}")
                print(f"   - Session summary chunks: {len([m for m in all_metadata if m['type'] == 'session_summaries'])}")
                print(f"   - Session report chunks: {len([m for m in all_metadata if m['type'] == 'session_report'])}")
                print(f"   - Therapy guideline chunks: {len([m for m in all_metadata if m['type'] == 'therapy_guidelines'])}")
            else:
                print(f"⚠️ No content found for patient {patient_id}")
                # Create empty knowledge base
                self.knowledge_base[patient_id] = {
                    'chunks': [],
                    'embeddings': None,
                    'metadata': [],
                    'vectorizer': None
                }
            
        except Exception as e:
            print(f"❌ Error building knowledge base: {e}")
            raise
    
    def _extract_patient_info(self, patient) -> str:
        """Extract patient information as text"""
        info_parts = [
            f"Patient ID: {patient.patientID}",
            f"Full Name: {patient.personalInfo.full_name}",
            f"Date of Birth: {patient.personalInfo.date_of_birth}",
            f"Gender: {patient.personalInfo.gender}",
            f"Occupation: {patient.personalInfo.occupation}",
            f"Marital Status: {patient.personalInfo.marital_status}",
        ]
        
        if patient.personalInfo.health_info:
            health_info = patient.personalInfo.health_info
            info_parts.extend([
                f"Current Medications: {health_info.current_medications}",
                f"Family History of Mental Illness: {health_info.family_history_of_mental_illness}",
                f"Physical Health Conditions: {health_info.physical_health_conditions}",
                f"Previous Diagnoses: {health_info.previous_diagnoses}",
                f"Substance Use: {health_info.substance_use}",
            ])
        
        if patient.personalInfo.therapy_info:
            therapy_info = patient.personalInfo.therapy_info
            info_parts.append(f"Reason for Therapy: {therapy_info.reason_for_therapy}")
        
        return "\n".join(info_parts)
    
    def _get_therapy_guidelines(self) -> str:
        """Return general therapy guidelines and best practices"""
        return """
        THERAPEUTIC GUIDELINES AND BEST PRACTICES:
        
        1. Active Listening: Always listen attentively and validate the patient's feelings.
        
        2. Confidentiality: Maintain strict confidentiality of all patient information.
        
        3. Non-judgmental Approach: Provide a safe, non-judgmental space for patients.
        
        4. Therapeutic Alliance: Build trust and rapport with the patient.
        
        5. Evidence-Based Practice: Use clinically proven therapeutic techniques.
        
        6. Crisis Assessment: Always assess for suicide risk and safety concerns.
        
        7. Progress Monitoring: Track patient progress and adjust treatment plans accordingly.
        
        8. Boundaries: Maintain appropriate professional boundaries.
        
        9. Cultural Sensitivity: Be aware of cultural factors affecting mental health.
        
        10. Documentation: Maintain accurate clinical records and session notes.
        
        COMMON THERAPEUTIC TECHNIQUES:
        - Cognitive Behavioral Therapy (CBT)
        - Dialectical Behavior Therapy (DBT)
        - Mindfulness-based interventions
        - Psychoeducation
        - Emotion regulation techniques
        - Grounding exercises for anxiety
        - Progressive muscle relaxation
        
        PROGRESS ASSESSMENT INDICATORS:
        - Improved emotional regulation
        - Better coping strategies
        - Increased self-awareness
        - Enhanced interpersonal relationships
        - Reduced symptom severity
        - Greater life satisfaction
        - Improved daily functioning
        """
    
    def retrieve_relevant_context(self, patient_id: int, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if patient_id not in self.knowledge_base:
            self.build_knowledge_base(patient_id)
        
        kb = self.knowledge_base[patient_id]
        
        # Check if we have any content
        if not kb['chunks'] or kb['embeddings'] is None:
            return []
        
        # Transform query using the same vectorizer
        query_embedding = kb['vectorizer'].transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, kb['embeddings'])[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include chunks with positive similarity
                relevant_chunks.append({
                    'text': kb['chunks'][idx],
                    'similarity': float(similarities[idx]),
                    'metadata': kb['metadata'][idx]
                })
        
        return relevant_chunks
    
    def generate_response(self, patient_id: int, query: str, chat_history: List[Dict] = None) -> str:
        """Generate response using RAG with enhanced cross-session analysis"""
        try:
            # Retrieve relevant context
            relevant_chunks = self.retrieve_relevant_context(patient_id, query)
            
            # Build context from retrieved chunks
            context_parts = []
            for chunk in relevant_chunks:
                metadata = chunk['metadata']
                source_info = f"[{metadata['type']} - {metadata['source']}]"
                context_parts.append(f"{source_info}\n{chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # If no relevant context found, use basic patient info
            if not context:
                try:
                    from model.patient_model import Patient
                    patient = Patient.objects(patientID=patient_id).first()
                    if patient:
                        context = self._extract_patient_info(patient)
                        # Also add session count info
                        context += f"\n\nThis patient has {len(patient.sessions)} therapy sessions on record."
                    else:
                        context = f"Patient {patient_id} information"
                except:
                    context = f"Patient {patient_id} information"
            
            # Build chat history context
            history_context = ""
            if chat_history:
                history_parts = []
                for msg in chat_history[-5:]:  # Last 5 messages
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    history_parts.append(f"{role.upper()}: {content}")
                history_context = "\n".join(history_parts)
            
            # Create prompt
            prompt = self._build_prompt(context, query, history_context, patient_id)
            
            # Generate response
            response = model.generate_content(prompt, generation_config={"temperature": 0.3})
            
            return response.text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble accessing the patient information right now. Please try again."
    
    def _build_prompt(self, context: str, query: str, history: str, patient_id: int) -> str:
        """Build the prompt for the LLM with enhanced cross-session analysis"""
        return f"""
You are an AI assistant specifically designed to help therapists analyze patient data and provide clinical insights. You have access to specific patient information, ALL session reports, and therapeutic guidelines.

STRICT GUIDELINES:
1. ONLY answer questions related to the provided patient data, session reports, or general therapeutic guidance
2. If asked about anything outside this context, politely redirect to patient-related topics
3. Maintain professional, clinical language appropriate for healthcare settings
4. Never provide medical diagnoses - only observations and suggestions for further clinical evaluation
5. Always prioritize patient safety and well-being
6. Respect patient confidentiality - only discuss information already provided in the context

CROSS-SESSION ANALYSIS CAPABILITIES:
- You have access to ALL therapy sessions for this patient
- You can track progress and patterns across multiple sessions
- You can identify improvement trends or concerning developments
- You can compare emotional states, engagement levels, and therapeutic responses over time
- You can assess the effectiveness of different therapeutic interventions
- You can provide session-by-session comparisons and chronological analysis

AVAILABLE CONTEXT FOR PATIENT {patient_id}:
{context}

RECENT CONVERSATION HISTORY:
{history}

THERAPIST QUERY: {query}

When analyzing patient progress:
1. Reference specific sessions when making comparisons (e.g., "Session 1 vs Session 3")
2. Note chronological improvements or deteriorations with specific examples
3. Identify patterns across multiple sessions
4. Suggest therapeutic adjustments based on session-to-session changes
5. Highlight any concerning trends that require immediate attention
6. Use session dates and numbers to provide concrete evidence
7. Quantify improvements where possible (e.g., "emotional regulation improved from Session 2 to Session 4")

Please provide a helpful, professional response based ONLY on the provided patient context and therapeutic guidelines. If the query is outside this scope, politely explain that you can only assist with this patient's clinical information and therapeutic guidance.

RESPONSE:
"""

    def clear_knowledge_base(self, patient_id: Optional[int] = None):
        """Clear knowledge base for specific patient or all patients"""
        if patient_id:
            self.knowledge_base.pop(patient_id, None)
            print(f"Knowledge base cleared for patient {patient_id}")
        else:
            self.knowledge_base.clear()
            print("All knowledge bases cleared")
    
    def get_knowledge_base_stats(self, patient_id: int) -> Dict:
        """Get statistics about the knowledge base for a patient"""
        if patient_id not in self.knowledge_base:
            return {"status": "not_built", "chunks": 0}
        
        kb = self.knowledge_base[patient_id]
        metadata_types = {}
        for meta in kb['metadata']:
            meta_type = meta['type']
            metadata_types[meta_type] = metadata_types.get(meta_type, 0) + 1
        
        # Get session count if available
        session_count = 0
        for meta in kb['metadata']:
            if meta['type'] == 'session_summaries' and 'total_sessions' in meta:
                session_count = meta['total_sessions']
                break
        
        return {
            "status": "built",
            "total_chunks": len(kb['chunks']),
            "chunk_types": metadata_types,
            "total_sessions": session_count,
            "last_updated": datetime.datetime.now().isoformat()
        }