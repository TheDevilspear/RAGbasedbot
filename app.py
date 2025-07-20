from flask import Flask, request, jsonify, session, redirect, render_template, url_for, g
from pymongo import MongoClient
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import os
import uuid
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader , Docx2txtLoader
import time


# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(16))

def utcnow():
    return datetime.now(timezone.utc)

# --- System Initialization ---
def initialize_mongodb():
    """Connects to MongoDB using the provided mongo_url."""
    mongo_url = os.environ.get("MONGO_URL")
    if not mongo_url:
        raise ValueError("MONGO_URL environment variable not set. Please provide MongoDB connection string.")
    
    client = MongoClient(mongo_url)
    try:
        client.admin.command('ismaster')
    except Exception as e:
        raise ConnectionError(f"Could not connect to MongoDB at {mongo_url}: {e}")
    
    return client["jamba_chatbot"]

def initialize_ai21_client():
    """Initializes the AI21 Labs Jamba-mini client using the API key."""
    api_key = os.environ.get("AI21_API_KEY")
    if not api_key:
        raise ValueError("AI21_API_KEY environment variable not set. Please provide your AI21 API key.")
    
    return AI21Client(api_key=api_key)

# Initialize DB and AI21 Client globally
try:
    db = initialize_mongodb()
    ai21_client = initialize_ai21_client()
    app.db = db
    app.ai21_client = ai21_client
except (ValueError, ConnectionError) as e:
    print(f"FATAL ERROR during app initialization: {e}")
    db = None
    ai21_client = None

# Load embedding model globally
try:
    embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    EMBEDDING_DIMENSIONS = embedding_model.get_sentence_embedding_dimension()
    embedding_model.eval()
    print(f"✅ Embedding model loaded successfully. Dimensions: {EMBEDDING_DIMENSIONS}")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None
    EMBEDDING_DIMENSIONS = 768  # Default for nomic-embed-text-v1

# Atlas Search Index Names - CORRECTED
VECTOR_SEARCH_INDEX_NAME = "vector_index"
FULL_TEXT_INDEX_NAME = "default"

app.config['VECTOR_SEARCH_INDEX_NAME'] = VECTOR_SEARCH_INDEX_NAME
app.config['FULL_TEXT_INDEX_NAME'] = FULL_TEXT_INDEX_NAME

# Load Speech-to-Text pipeline
try:
    from transformers import pipeline
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")
except Exception as e:
    print(f"Error loading Whisper ASR model: {e}")
    stt_pipeline = None

# Create uploads directory
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Helper Functions ---
def embed_text(text: str) -> list[float]:
    """
    Embeds text using the nomic-embed-text-v1 model with better error handling.
    """
    if not embedding_model:
        print("Embedding model not loaded. Cannot embed text.")
        return None

    if not text or not text.strip():
        print("Empty text provided for embedding")
        return None

    try:
        # Normalize text
        text = text.strip()
        if len(text) > 8000:  # Arbitrary limit to prevent memory issues
            print(f"Text too long ({len(text)} chars), truncating")
            text = text[:8000]
        
        embedding = embedding_model.encode([text], normalize_embeddings=True)
        
        if embedding is None or len(embedding) == 0:
            print("Embedding model returned None or empty result")
            return None
            
        result = embedding[0].tolist()
        
        # Validate embedding
        if not isinstance(result, list) or len(result) == 0:
            print("Invalid embedding format")
            return None
            
        return result
        
    except Exception as e:
        print(f"Error during embedding: {e}")
        return None

def chunk_document_langchain(document_path: str, file_extension: str):
    """
    Loads document and chunks it using RecursiveCharacterTextSplitter.
    Fixed to handle multiple file types properly.
    """
    documents = []
    try:
        print(f"Processing document: {document_path} with extension: {file_extension}")
        
        if file_extension.lower() == "pdf":
            loader = PyPDFLoader(document_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            
        elif file_extension.lower() == "txt":
            loader = TextLoader(document_path, encoding='utf-8')
            documents = loader.load()
            print(f"Loaded {len(documents)} text documents")
            
        elif file_extension.lower() in ["docx", "doc"]:
            # Use Docx2txtLoader for Word documents
            loader = Docx2txtLoader(document_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from Word file")
            
        else:
            print(f"Unsupported file type: {file_extension}")
            return []
            
    except Exception as e:
        print(f"Error loading document with extension {file_extension}: {e}")
        return []

    if not documents:
        print("No documents were loaded")
        return []

    # Check if documents have content
    total_content_length = sum(len(doc.page_content) for doc in documents)
    print(f"Total content length: {total_content_length} characters")
    
    if total_content_length == 0:
        print("Documents loaded but contain no text content")
        return []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    try:
        final_chunks = text_splitter.split_documents(documents)
        print(f"Created {len(final_chunks)} chunks from documents")
        
        # Extract text content and filter out empty chunks
        chunk_texts = []
        for i, chunk in enumerate(final_chunks):
            chunk_text = chunk.page_content.strip()
            if chunk_text:  # Only add non-empty chunks
                chunk_texts.append(chunk_text)
                print(f"Chunk {i+1}: {len(chunk_text)} characters")
        
        print(f"Final chunk count after filtering: {len(chunk_texts)}")
        return chunk_texts
        
    except Exception as e:
        print(f"Error during text splitting: {e}")
        return []


def generate_chat_name_from_messages(user_message, bot_message):
    """Auto-generates a chat name using keywords from the first user message."""
    if user_message:
        return "Chat about " + " ".join(user_message.split()[:5]) + "..."
    return "Untitled Chat"

def _convert_history_for_ai21(raw_messages):
    """Converts stored messages to AI21 Chat API format."""
    out = []
    for m in raw_messages:
        role = m["role"]
        if role == "bot":
            role = "assistant"
        elif role not in ("user", "assistant", "system"):
            role = "user"
        out.append(ChatMessage(content=m["content"], role=role))
    return out

# --- CORRECTED INDEX CREATION FUNCTIONS ---
def create_atlas_vector_index():
    """Creates the Atlas Vector Search index with correct syntax."""
    if db is None:
        print("MongoDB connection not initialized. Cannot create vector index.")
        return False
    
    print(f"🔍 Creating Atlas Vector Search index: '{VECTOR_SEARCH_INDEX_NAME}'")
    
    # CORRECTED: Proper vector search index definition
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": EMBEDDING_DIMENSIONS,
                    "path": "embedding",
                    "similarity": "cosine"
                },
                {
                    "type": "filter",
                    "path": "user_email"
                },
                {
                    "type": "filter",
                    "path": "doc_id"
                }
            ]
        },
        name=VECTOR_SEARCH_INDEX_NAME,
        type="vectorSearch"
    )
    
    try:
        existing_indices = list(db.document_chunks.list_search_indexes(name=VECTOR_SEARCH_INDEX_NAME))
        if not existing_indices:
            db.document_chunks.create_search_index(model=search_index_model)
            print("🚀 Vector index creation request sent. Waiting for readiness...")
            
            # Wait for index to be ready
            while True:
                indices = list(db.document_chunks.list_search_indexes(VECTOR_SEARCH_INDEX_NAME))
                if len(indices) and indices[0].get("queryable") is True:
                    print(f"✅ Vector index '{VECTOR_SEARCH_INDEX_NAME}' is ready.")
                    return True
                print("⏳ Waiting for vector index...")
                time.sleep(5)
        else:
            print(f"ℹ️ Vector index '{VECTOR_SEARCH_INDEX_NAME}' already exists.")
            return True
    except Exception as e:
        print(f"❌ Error creating vector search index: {e}")
        return False

def create_atlas_full_text_index():
    """Creates the Atlas Search (full-text) index with correct syntax."""
    if db is None:
        print("MongoDB connection not initialized. Cannot create full-text index.")
        return False

    print(f"🔍 Creating Atlas Full-Text Search index: '{FULL_TEXT_INDEX_NAME}'")
    
    # CORRECTED: Proper full-text search index definition
    search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {
                        "type": "text",
                        "analyzer": "lucene.english"
                    },
                    "user_email": {
                        "type": "token"
                    },
                    "doc_id": {
                        "type": "token"
                    }
                }
            }
        },
        name=FULL_TEXT_INDEX_NAME,
        type="search"
    )
    
    try:
        existing_indices = list(db.document_chunks.list_search_indexes(name=FULL_TEXT_INDEX_NAME))
        if not existing_indices:
            db.document_chunks.create_search_index(model=search_index_model)
            print("🚀 Full-text index creation request sent. Waiting for readiness...")
            
            # Wait for index to be ready
            while True:
                indices = list(db.document_chunks.list_search_indexes(FULL_TEXT_INDEX_NAME))
                if len(indices) and indices[0].get("queryable") is True:
                    print(f"✅ Full-text index '{FULL_TEXT_INDEX_NAME}' is ready.")
                    return True
                print("⏳ Waiting for full-text index...")
                time.sleep(5)
        else:
            print(f"ℹ️ Full-text index '{FULL_TEXT_INDEX_NAME}' already exists.")
            return True
    except Exception as e:
        print(f"❌ Error creating full-text search index: {e}")
        return False

# --- CORRECTED HYBRID SEARCH IMPLEMENTATION ---
def perform_hybrid_search(user_message: str, query_embedding: list, email: str, selected_doc_ids: list, limit: int = 5):
    """
    Performs hybrid search using separate vector and text searches with post-filtering.
    """
    if not query_embedding:
        print("No query embedding provided for hybrid search")
        return []
    
    try:
        # Step 1: Vector Search Pipeline (using vector index)
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_SEARCH_INDEX_NAME,  # This should be your vector index name
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": 50  # Get more results to filter later
                }
            },
            {
                "$match": {
                    "user_email": email,
                    "doc_id": {"$in": selected_doc_ids}
                }
            },
            {
                "$project": {
                    "text": 1,
                    "doc_id": 1,
                    "user_email": 1,
                    "vector_score": {"$meta": "vectorSearchScore"},
                    "_id": 0
                }
            },
            {
                "$limit": 20
            }
        ]
        
        # Step 2: Full-Text Search Pipeline (using search index)
        text_pipeline = [
            {
                "$search": {
                    "index": FULL_TEXT_INDEX_NAME,  # This should be your search index name
                    "compound": {
                        "must": [
                            {
                                "text": {
                                    "query": user_message,
                                    "path": "text"
                                }
                            }
                        ],
                        "filter": [
                            {
                                "equals": {
                                    "path": "user_email",
                                    "value": email
                                }
                            },
                            {
                                "in": {
                                    "path": "doc_id",
                                    "value": selected_doc_ids
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$project": {
                    "text": 1,
                    "doc_id": 1,
                    "user_email": 1,
                    "text_score": {"$meta": "searchScore"},
                    "_id": 0
                }
            },
            {
                "$limit": 20
            }
        ]
        
        # Execute both searches
        vector_results = list(db.document_chunks.aggregate(vector_pipeline))
        text_results = list(db.document_chunks.aggregate(text_pipeline))
        
        # Step 3: Reciprocal Rank Fusion (RRF)
        rrf_results = reciprocal_rank_fusion(vector_results, text_results, limit)
        
        return rrf_results
        
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        return []


def reciprocal_rank_fusion(vector_results: list, text_results: list, limit: int, k: int = 60):
    """
    Implements Reciprocal Rank Fusion to combine vector and text search results.
    RRF Score = 1/(k + rank) for each result, summed across all search methods.
    """
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Process vector search results
    for rank, doc in enumerate(vector_results, 1):
        doc_key = doc["text"]  # Use text as unique identifier
        rrf_score = 1.0 / (k + rank)
        combined_scores[doc_key] = {
            "document": doc,
            "rrf_score": rrf_score,
            "vector_rank": rank,
            "text_rank": None
        }
    
    # Process text search results
    for rank, doc in enumerate(text_results, 1):
        doc_key = doc["text"]
        rrf_score = 1.0 / (k + rank)
        
        if doc_key in combined_scores:
            # Document found in both searches - add scores
            combined_scores[doc_key]["rrf_score"] += rrf_score
            combined_scores[doc_key]["text_rank"] = rank
        else:
            # Document only in text search
            combined_scores[doc_key] = {
                "document": doc,
                "rrf_score": rrf_score,
                "vector_rank": None,
                "text_rank": rank
            }
    
    # Sort by RRF score (descending) and take top results
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    # Return top documents with their scores
    final_results = []
    for item in sorted_results[:limit]:
        doc = item["document"]
        doc["rrf_score"] = item["rrf_score"]
        doc["vector_rank"] = item["vector_rank"]
        doc["text_rank"] = item["text_rank"]
        final_results.append(doc)
    
    return final_results
def reciprocal_rank_fusion(vector_results: list, text_results: list, limit: int, k: int = 60):
    """
    Implements Reciprocal Rank Fusion to combine vector and text search results.
    RRF Score = 1/(k + rank) for each result, summed across all search methods.
    """
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Process vector search results
    for rank, doc in enumerate(vector_results, 1):
        doc_key = doc["text"]  # Use text as unique identifier
        rrf_score = 1.0 / (k + rank)
        combined_scores[doc_key] = {
            "document": doc,
            "rrf_score": rrf_score,
            "vector_rank": rank,
            "text_rank": None
        }
    
    # Process text search results
    for rank, doc in enumerate(text_results, 1):
        doc_key = doc["text"]
        rrf_score = 1.0 / (k + rank)
        
        if doc_key in combined_scores:
            # Document found in both searches - add scores
            combined_scores[doc_key]["rrf_score"] += rrf_score
            combined_scores[doc_key]["text_rank"] = rank
        else:
            # Document only in text search
            combined_scores[doc_key] = {
                "document": doc,
                "rrf_score": rrf_score,
                "vector_rank": None,
                "text_rank": rank
            }
    
    # Sort by RRF score (descending) and take top results
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    # Return top documents with their scores
    final_results = []
    for item in sorted_results[:limit]:
        doc = item["document"]
        doc["rrf_score"] = item["rrf_score"]
        doc["vector_rank"] = item["vector_rank"]
        doc["text_rank"] = item["text_rank"]
        final_results.append(doc)
    
    return final_results

# --- User Authentication & Session ---
@app.route('/')
def index():
    """Redirects the root URL to the login page."""
    return redirect(url_for('login_user'))

@app.route("/login", methods=["GET", "POST"])
def login_user():
    """Accepts user email, creates a session, and stores user in `users` collection."""
    if request.method == "POST":
        email = request.form.get("email")
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        session["email"] = email
        if not db.users.find_one({"email": email}):
            db.users.insert_one({"email": email, "created_at": utcnow()})
        return redirect("/dashboard")
    return render_template("login.html")

@app.route("/logout")
def logout_user():
    """Clears session and redirects to login page."""
    session.clear()
    return redirect("/login")

# --- Dashboard ---
@app.route("/dashboard")
def load_user_dashboard():
    """Loads the main interface with chat list, document list, and current chat."""
    email = session.get("email")
    if not email:
        return redirect("/login")

    chats = list(db.conversations.find({"user_email": email}).sort("updated_at", -1))
    documents = list(db.document_metadata.find({"user_email": email}))

    current_chat_id = request.args.get("chat_id")
    current_chat = None
    if current_chat_id:
        current_chat = db.conversations.find_one({"chat_id": current_chat_id, "user_email": email})
    elif chats:
        current_chat = chats[0]

    return render_template("dashboard.html", chats=chats, documents=documents, current_chat=current_chat)

@app.route("/resume_chat/<chat_id>")
def resume_chat(chat_id):
    """Loads a previous chat session."""
    email = session.get("email")
    if not email:
        return redirect("/login")

    chat = db.conversations.find_one({"chat_id": chat_id, "user_email": email})
    if chat:
        return redirect(f"/dashboard?chat_id={chat_id}")
    return redirect("/dashboard")

# --- Chat Management ---
@app.route("/start_chat", methods=["POST"])
def start_new_chat():
    """Creates a new chat session."""
    email = session.get("email")
    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    chat_id = str(uuid.uuid4())
    new_chat_session = {
        "chat_id": chat_id,
        "user_email": email,
        "chat_name": "New Chat",
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "mode": "general",
        "selected_documents": [],
        "messages": []
    }
    db.conversations.insert_one(new_chat_session)
    return jsonify({"chat_id": chat_id, "status": "success"})

@app.route("/rename_chat", methods=["POST"])
def rename_chat():
    """Updates the name of a chat session."""
    chat_id = request.form.get("chat_id")
    new_name = request.form.get("new_name")
    if not chat_id or not new_name:
        return jsonify({"error": "Chat ID and new name are required"}), 400

    db.conversations.update_one(
        {"chat_id": chat_id, "user_email": session.get("email")},
        {"$set": {"chat_name": new_name, "updated_at": utcnow()}}
    )
    return jsonify({"status": "success"})

@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    """Permanently deletes a chat session."""
    chat_id = request.form.get("chat_id")
    if not chat_id:
        return jsonify({"error": "Chat ID is required"}), 400

    db.conversations.delete_one({"chat_id": chat_id, "user_email": session.get("email")})
    return jsonify({"status": "deleted"})

@app.route("/switch_chat_mode", methods=["POST"])
def switch_chat_mode():
    """Toggles between general and RAG mode."""
    chat_id = request.form.get("chat_id")
    new_mode = request.form.get("mode")
    if not chat_id or new_mode not in ["general", "document"]:
        return jsonify({"error": "Invalid chat ID or mode"}), 400

    db.conversations.update_one(
        {"chat_id": chat_id, "user_email": session.get("email")},
        {"$set": {"mode": new_mode, "updated_at": utcnow()}}
    )
    return jsonify({"status": "mode updated"})

# --- Performance monitoring ---
@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        print(f"Request took {duration:.2f} seconds")
    return response

# --- CORRECTED CHAT INTERACTION ---
@app.route("/send_message", methods=["POST"])
def send_message():
    """Handles user message input with corrected hybrid search."""
    chat_id = request.form.get("chat_id")
    user_message = request.form.get("message")
    email = session.get("email")

    if not email:
        return jsonify({"error": "Unauthorized"}), 401
    if not chat_id or not user_message:
        return jsonify({"error": "Chat ID and message are required"}), 400

    chat = db.conversations.find_one({"chat_id": chat_id, "user_email": email})
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    mode = chat["mode"]
    history = chat["messages"]
    
    ai21_messages = _convert_history_for_ai21(history)
    ai21_messages.append(ChatMessage(content=user_message, role="user"))

    system_prompt_general = (
        "You are a helpful, respectful, and honest assistant. "
        "Always be safe, and do not generate content that is harmful, hateful, "
        "illegal, or violent. "
        "If you are unsure how to respond to a query, or if the information "
        "you are asked to recall is not within the scope of a general assistant, "
        "you can politely state that you cannot fulfill the request or that you "
        "do not have that specific information."
    )
    
    bot_reply = "Sorry, I couldn't generate a response due to an internal error."

    try:
        if mode == "document":
            selected_doc_ids = chat.get("selected_documents", [])
            if not selected_doc_ids:
                return jsonify({"reply": "Please select documents to use RAG mode."}), 400

            query_embedding = embed_text(user_message)
            if query_embedding is None:
                return jsonify({"reply": "Failed to create embedding for your query."}), 500

            # Use the corrected hybrid search
            start_search_time = time.time()
            top_chunks = perform_hybrid_search(
                user_message=user_message,
                query_embedding=query_embedding,
                email=email,
                selected_doc_ids=selected_doc_ids,
                limit=5
            )
            search_time = time.time() - start_search_time
            print(f"Hybrid search took {search_time:.3f} seconds")

            if not top_chunks:
                return jsonify({"reply": "No relevant information found in the selected documents."}), 200

            # Create context from search results
            context_parts = []
            for chunk in top_chunks:
                context_parts.append(f"[Score: {chunk.get('rrf_score', 0):.4f}] {chunk['text']}")
            
            context = "\n\n".join(context_parts)

            # RAG-specific system prompt
            system_prompt_rag = (
                "You are a helpful assistant. Only answer using the information provided in the DOCUMENTS section below. "
                "If the answer is not found in the documents, respond with: 'I'm sorry, the information you're asking for is not available in the provided documents.'\n\n"
                "DOCUMENTS:\n" + context
            )
            ai21_messages.insert(0, ChatMessage(content=system_prompt_rag, role="system"))
        else:
            ai21_messages.insert(0, ChatMessage(content=system_prompt_general, role="system"))

        # Generate response
        response = ai21_client.chat.completions.create(
            messages=ai21_messages,
            model="jamba-mini-1.6-2025-03"
        )
        bot_reply = response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling AI21 or during RAG process: {e}")
        bot_reply = "Sorry, I couldn't generate a response. Please try again later."
        return jsonify({"reply": bot_reply}), 500

    # Update conversation
    try:
        db.conversations.update_one(
            {"chat_id": chat_id, "user_email": email},
            {
                "$push": {"messages": {"$each": [
                    {"role": "user", "content": user_message, "timestamp": utcnow()},
                    {"role": "bot", "content": bot_reply, "timestamp": utcnow()}
                ]}},
                "$set": {"updated_at": utcnow()}
            }
        )
    except PyMongoError as e:
        print(f"Database update error: {e}")
        pass

    # Auto-generate chat name if needed
    if chat["chat_name"] == "New Chat" and len(chat["messages"]) == 0:
        try:
            generated_name = generate_chat_name_from_messages(user_message, bot_reply)
            db.conversations.update_one(
                {"chat_id": chat_id, "user_email": email},
                {"$set": {"chat_name": generated_name, "updated_at": utcnow()}}
            )
        except Exception as e:
            print(f"Error generating chat name: {e}")
            pass

    return jsonify({"reply": bot_reply, "chat_id": chat_id})

# --- Document Management ---
@app.route("/upload_document", methods=["POST"])
def upload_document():
    """
    Fixed document upload with better error handling and validation.
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        chat_id = request.form.get("chat_id")
        if not chat_id:
            return jsonify({"error": "chat_id is required"}), 400

        email = session.get("email")
        if not email:
            return jsonify({"error": "Unauthorized"}), 401

        # Validate file type
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
            
        file_extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        allowed_extensions = ['pdf', 'txt', 'docx', 'doc']
        
        if file_extension not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"}), 400

        # Check file size (optional - already handled by Flask config)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size == 0:
            return jsonify({"error": "File is empty"}), 400

        print(f"Processing file: {filename} ({file_size} bytes)")

        # Save file temporarily
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({"error": "Failed to save uploaded file"}), 500

        # Process document
        try:
            final_chunks_text_content = chunk_document_langchain(filepath, file_extension)
        except Exception as e:
            print(f"Error processing document: {e}")
            return jsonify({"error": f"Failed to process document: {str(e)}"}), 500
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Temporary file removed: {filepath}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {filepath}: {e}")

        if not final_chunks_text_content:
            return jsonify({"error": "No text content could be extracted from the document"}), 400

        print(f"Successfully extracted {len(final_chunks_text_content)} chunks")

        # Create embeddings and store chunks
        doc_id = str(uuid.uuid4())
        stored_chunks_count = 0
        chunks_to_insert = []
        failed_embeddings = 0
        
        for i, chunk_text_content in enumerate(final_chunks_text_content):
            if not chunk_text_content.strip():
                continue
                
            print(f"Processing chunk {i+1}/{len(final_chunks_text_content)}")
            embedding = embed_text(chunk_text_content)
            
            if embedding is not None:
                chunks_to_insert.append({
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "user_email": email,
                    "text": chunk_text_content,
                    "embedding": embedding,
                    "created_at": utcnow()
                })
                stored_chunks_count += 1
            else:
                failed_embeddings += 1
                print(f"Failed to create embedding for chunk {i+1}")

        if stored_chunks_count == 0:
            return jsonify({"error": "No chunks could be processed and embedded"}), 500

        if failed_embeddings > 0:
            print(f"Warning: {failed_embeddings} chunks failed to embed")

        # Store chunks in database
        try:
            result = db.document_chunks.insert_many(chunks_to_insert)
            print(f"Inserted {len(result.inserted_ids)} chunks into database")
        except Exception as e:
            print(f"Error inserting chunks into database: {e}")
            return jsonify({"error": "Failed to store document chunks in database"}), 500

        # Store document metadata
        try:
            metadata_doc = {
                "doc_id": doc_id,
                "user_email": email,
                "original_name": filename,
                "current_name": filename,
                "file_extension": file_extension,
                "file_size": file_size,
                "chunks_count": stored_chunks_count,
                "uploaded_at": utcnow()
            }
            db.document_metadata.insert_one(metadata_doc)
            print(f"Document metadata stored for doc_id: {doc_id}")
        except Exception as e:
            print(f"Error storing document metadata: {e}")
            # Try to clean up chunks if metadata fails
            try:
                db.document_chunks.delete_many({"doc_id": doc_id})
            except:
                pass
            return jsonify({"error": "Failed to store document metadata"}), 500

        # Add to chat's selected documents
        try:
            result = db.conversations.update_one(
                {"chat_id": chat_id, "user_email": email},
                {
                    "$addToSet": {"selected_documents": doc_id},
                    "$set": {"updated_at": utcnow()}
                }
            )
            if result.matched_count == 0:
                print(f"Warning: Chat {chat_id} not found when adding document")
            else:
                print(f"Document {doc_id} added to chat {chat_id}")
        except Exception as e:
            print(f"Error adding document to chat: {e}")
            # Don't fail the entire operation for this

        return jsonify({
            "status": "uploaded",
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": stored_chunks_count,
            "failed_embeddings": failed_embeddings
        })

    except Exception as e:
        print(f"Unexpected error in upload_document: {e}")
        return jsonify({"error": "An unexpected error occurred during file upload"}), 500

@app.route("/rename_document", methods=["POST"])
def rename_document():
    """Updates the name of a document."""
    doc_id = request.form.get("doc_id")
    new_name = request.form.get("new_name")
    if not doc_id or not new_name:
        return jsonify({"error": "Document ID and new name are required"}), 400

    db.document_metadata.update_one(
        {"doc_id": doc_id, "user_email": session.get("email")},
        {"$set": {"current_name": new_name}}
    )
    return jsonify({"status": "renamed"})

@app.route("/delete_document", methods=["POST"])
def delete_document():
    """Deletes a document and its associated chunks."""
    doc_id = request.form.get("doc_id")
    if not doc_id:
        return jsonify({"error": "Document ID is required"}), 400

    user_email = session.get("email")
    if not user_email:
        return jsonify({"error": "Unauthorized"}), 401

    db.document_metadata.delete_one({"doc_id": doc_id, "user_email": user_email})
    db.document_chunks.delete_many({"doc_id": doc_id, "user_email": user_email})
    
    db.conversations.update_many(
        {"user_email": user_email},
        {"$pull": {"selected_documents": doc_id}}
    )
    return jsonify({"status": "deleted"})


@app.route("/toggle_document", methods=["POST"])
def toggle_document_selection():
    """
    Adds or removes a document ID from the `selected_documents` list in a chat session.
    """
    chat_id = request.form.get("chat_id")
    doc_id = request.form.get("doc_id")
    action = request.form.get("action")
    email = session.get("email")

    if not chat_id or not doc_id or action not in ["select", "deselect"]:
        return jsonify({"error": "Chat ID, Document ID, and valid action ('select'/'deselect') are required."}), 400
    if not email:
        return jsonify({"error": "Unauthorized"}), 401

    if action == "select":
        db.conversations.update_one(
            {"chat_id": chat_id, "user_email": email},
            {"$addToSet": {"selected_documents": doc_id}, "$set": {"updated_at": utcnow()}}
        )
    else:
        db.conversations.update_one(
            {"chat_id": chat_id, "user_email": email},
            {"$pull": {"selected_documents": doc_id}, "$set": {"updated_at": utcnow()}}
        )
    return jsonify({"status": "updated"})

# --- Speech Input Functions ---
@app.route("/speech_input", methods=["POST"])
def speech_input():
    """
    Accepts live audio input from the frontend, transcribes it, and sends as a user message.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400
    audio_file = request.files["audio"]
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file"}), 400

    if not stt_pipeline:
        return jsonify({"error": "Speech recognition model not loaded."}), 503

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
    audio_file.save(filepath)

    transcribed_text = "Sorry, speech recognition failed."
    try:
        result = stt_pipeline(filepath)
        transcribed_text = result["text"]
    except Exception as e:
        print(f"Error during speech transcription: {e}")
        transcribed_text = "Sorry, I could not process the audio."
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({"transcribed_text": transcribed_text})

# --- Run the app ---
if __name__ == "__main__":
    # Ensure MongoDB and AI21 clients are initialized
    if db is None or ai21_client is None:
        print("Application cannot start due to initialization errors. Please check previous error messages.")
        exit(1) # Exit if essential services are not available

    # --- Create/Verify Atlas Search Indexes ---
    # These functions will attempt to create the indexes if they don't exist.
    # They use the VECTOR_SEARCH_INDEX_NAME and FULL_TEXT_INDEX_NAME variables defined globally.
    
    
    app.run(debug=True, port=5000)