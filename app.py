from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os
import threading
import time
from werkzeug.utils import secure_filename
import json
import requests
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


mongo_url = os.environ.get("MONGO_URL")

# MongoDB connection
try:
    client = MongoClient(mongo_url)
    db = client['jamba_chatbot']
    users_collection = db['users']
    conversations_collection = db['conversations']
    documents_collection = db['documents']
    document_metadata_collection = db['document_metadata']
    document_chunks_collection = db['document_chunks']
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"MongoDB connection error: {e}")

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded successfully!")

# AI21 Configuration (mock for demo - replace with actual API key)
AI21_API_KEY =  os.environ.get("AI21_API_KEY")
AI21_BASE_URL = "https://api.ai21.com/studio/v1"

def serialize_doc(doc):
    """Convert MongoDB document to JSON serializable format"""
    if doc is None:
        return None
    doc['_id'] = str(doc['_id'])
    return doc

def get_ai21_response(messages, mode="global", context=""):
    """Get response from AI21 Jamba-Mini model (mock implementation)"""
    try:
        # This is a mock implementation - replace with actual AI21 API call
        # For demo purposes, we'll simulate responses
        if mode == "rag" and context:
            # RAG mode response
            user_message = messages[-1]['content'] if messages else ""
            mock_response = f"Based on the provided documents, I can tell you about: {user_message}. [This is a mock response - integrate with actual AI21 API]"
        else:
            # Global mode response
            user_message = messages[-1]['content'] if messages else ""
            mock_response = f"I understand you're asking about: {user_message}. [This is a mock response - integrate with actual AI21 API]"
        
        return mock_response
    except Exception as e:
        print(f"AI21 API error: {e}")
        return "I'm sorry, I'm having trouble processing your request right now."

def extract_text_from_file(file_path, file_type):
    """Extract text from uploaded file"""
    text = ""
    try:
        if file_type == 'pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        elif file_type in ['doc', 'docx']:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text

def chunk_text(text, chunk_size=512):
    """Chunk text into smaller pieces"""
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "."
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_document_background(doc_id, file_path, filename, file_type, user_email):
    """Background process for document processing"""
    try:
        # Update status to processing
        documents_collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"processing_status": "processing"}}
        )
        
        # Extract text
        text = extract_text_from_file(file_path, file_type)
        
        # Chunk text
        chunks = chunk_text(text)
        
        # Generate embeddings and store chunks
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            
            chunk_doc = {
                "document_id": ObjectId(doc_id),
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding,
                "token_count": len(chunk.split()),
                "created_at": datetime.utcnow()
            }
            document_chunks_collection.insert_one(chunk_doc)
        
        # Generate title using first chunk
        title = generate_document_title(chunks[0] if chunks else filename)
        
        # Update document metadata
        metadata = {
            "document_id": ObjectId(doc_id),
            "total_chunks": len(chunks),
            "processing_time": 0,  # You can track this
            "file_size": os.path.getsize(file_path),
            "chunk_overlap": 0,
            "embedding_model": "all-MiniLM-L6-v2"
        }
        document_metadata_collection.insert_one(metadata)
        
        # Update document status
        documents_collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {
                "processing_status": "completed",
                "model_generated_title": title,
                "title": title
            }}
        )
        
        # Clean up file
        os.remove(file_path)
        
        print(f"Document {doc_id} processed successfully!")
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        documents_collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"processing_status": "error"}}
        )

def generate_document_title(text):
    """Generate title for document using AI21 (mock implementation)"""
    try:
        # Mock implementation - replace with actual AI21 call
        first_words = text.split()[:10]
        title = " ".join(first_words) + "..."
        return title[:50]  # Limit title length
    except:
        return "Untitled Document"

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# User Management
@app.route('/api/users', methods=['POST'])
def create_or_get_user():
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        # Check if user exists
        user = users_collection.find_one({"email": email})
        
        if not user:
            # Create new user
            user_doc = {
                "email": email,
                "created_at": datetime.utcnow()
            }
            result = users_collection.insert_one(user_doc)
            user_doc['_id'] = result.inserted_id
            user = user_doc
        
        return jsonify(serialize_doc(user))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<user_email>/conversations', methods=['GET'])
def get_user_conversations(user_email):
    try:
        conversations = list(conversations_collection.find(
            {"user_email": user_email}
        ).sort("updated_at", -1))
        
        return jsonify([serialize_doc(conv) for conv in conversations])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Conversation Management
@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    try:
        data = request.get_json()
        user_email = data.get('user_email')
        title = data.get('title', f"Untitled conversation on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
        
        if not user_email:
            return jsonify({"error": "User email is required"}), 400
        
        conv_doc = {
            "user_email": user_email,
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "current_mode": "global",
            "messages": []
        }
        
        result = conversations_collection.insert_one(conv_doc)
        conv_doc['_id'] = result.inserted_id
        
        return jsonify(serialize_doc(conv_doc))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    try:
        conversation = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify(serialize_doc(conversation))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations/<conversation_id>/rename', methods=['PUT'])
def rename_conversation(conversation_id):
    try:
        data = request.get_json()
        new_title = data.get('new_title')
        
        if not new_title:
            return jsonify({"error": "New title is required"}), 400
        
        result = conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"title": new_title}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"status": "success", "message": "Conversation renamed", "new_title": new_title})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    try:
        result = conversations_collection.delete_one({"_id": ObjectId(conversation_id)})
        
        if result.deleted_count == 0:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"status": "success", "message": "Conversation deleted"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations/<conversation_id>/mode', methods=['POST'])
def switch_mode(conversation_id):
    try:
        data = request.get_json()
        mode = data.get('mode')
        
        if mode not in ['global', 'rag']:
            return jsonify({"error": "Invalid mode"}), 400
        
        result = conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"current_mode": mode}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"status": "success", "current_mode": mode})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Chat functionality
@app.route('/api/chat/<conversation_id>', methods=['POST'])
def chat(conversation_id):
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get conversation
        conversation = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        
        # Add user message
        user_msg = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow()
        }
        
        # Get AI response based on mode
        messages = conversation['messages'] + [user_msg]
        mode = conversation.get('current_mode', 'global')
        
        if mode == 'rag':
            # RAG mode: search for relevant documents
            user_email = conversation['user_email']
            context = get_rag_context(user_message, user_email)
            response = get_ai21_response(messages, mode="rag", context=context)
        else:
            # Global mode
            response = get_ai21_response(messages, mode="global")
        
        # Add bot response
        bot_msg = {
            "role": "bot",
            "content": response,
            "timestamp": datetime.utcnow()
        }
        
        # Update conversation
        conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$push": {"messages": {"$each": [user_msg, bot_msg]}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return jsonify({
            "response": response,
            "conversation_id": conversation_id,
            "new_messages": [user_msg, bot_msg]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_rag_context(query, user_email):
    """Get relevant context from active documents"""
    try:
        # Get active documents for user
        active_docs = list(documents_collection.find({
            "user_email": user_email,
            "is_active": True,
            "processing_status": "completed"
        }))
        
        if not active_docs:
            return ""
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search for relevant chunks
        relevant_chunks = []
        for doc in active_docs:
            chunks = list(document_chunks_collection.find({"document_id": doc['_id']}))
            
            for chunk in chunks:
                # Calculate similarity
                similarity = cosine_similarity(
                    [query_embedding],
                    [chunk['embedding']]
                )[0][0]
                
                if similarity > 0.3:  # Threshold for relevance
                    relevant_chunks.append({
                        "content": chunk['content'],
                        "similarity": similarity
                    })
        
        # Sort by similarity and take top chunks
        relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks[:3]])
        
        return context
    
    except Exception as e:
        print(f"Error getting RAG context: {e}")
        return ""

# Document Management
@app.route('/api/documents', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        user_email = request.form.get('user_email')
        
        if not user_email:
            return jsonify({"error": "User email is required"}), 400
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        filename = secure_filename(file.filename)
        file_type = filename.split('.')[-1].lower()
        
        if file_type not in ['pdf', 'doc', 'docx', 'txt']:
            return jsonify({"error": "Invalid file type"}), 400
        
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Create document record
        doc_record = {
            "user_email": user_email,
            "title": filename,
            "filename": filename,
            "file_type": file_type,
            "is_active": False,
            "processing_status": "pending",
            "created_at": datetime.utcnow(),
            "model_generated_title": ""
        }
        
        result = documents_collection.insert_one(doc_record)
        doc_id = str(result.inserted_id)
        
        # Start background processing
        thread = threading.Thread(
            target=process_document_background,
            args=(doc_id, file_path, filename, file_type, user_email)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "processing",
            "document_id": doc_id,
            "filename": filename
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    try:
        user_email = request.args.get('user_email')
        if not user_email:
            return jsonify({"error": "User email is required"}), 400
        
        documents = list(documents_collection.find(
            {"user_email": user_email}
        ).sort("created_at", -1))
        
        return jsonify([serialize_doc(doc) for doc in documents])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<document_id>/activate', methods=['PUT'])
def activate_document(document_id):
    try:
        result = documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"is_active": True}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Document not found"}), 404
        
        return jsonify({"status": "success", "message": "Document activated"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<document_id>/inactivate', methods=['PUT'])
def inactivate_document(document_id):
    try:
        result = documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"is_active": False}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Document not found"}), 404
        
        return jsonify({"status": "success", "message": "Document inactivated"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<document_id>/rename', methods=['PUT'])
def rename_document(document_id):
    try:
        data = request.get_json()
        new_title = data.get('new_title')
        
        if not new_title:
            return jsonify({"error": "New title is required"}), 400
        
        result = documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"title": new_title}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Document not found"}), 404
        
        return jsonify({"status": "success", "message": "Document renamed"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    try:
        # Delete from all collections
        documents_collection.delete_one({"_id": ObjectId(document_id)})
        document_metadata_collection.delete_one({"document_id": ObjectId(document_id)})
        document_chunks_collection.delete_many({"document_id": ObjectId(document_id)})
        
        return jsonify({"status": "success", "message": "Document deleted"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<document_id>/status', methods=['GET'])
def get_document_status(document_id):
    try:
        document = documents_collection.find_one({"_id": ObjectId(document_id)})
        
        if not document:
            return jsonify({"error": "Document not found"}), 404
        
        return jsonify({
            "status": document.get('processing_status', 'unknown'),
            "document_id": document_id
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)