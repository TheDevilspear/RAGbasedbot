import io
from flask import Flask, request, jsonify, send_file, session, redirect, render_template, url_for,g
from gtts import gTTS
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
from langchain_community.document_loaders import TextLoader , Docx2txtLoader
import time
import pytesseract
from PIL import Image ,ImageEnhance, ImageFilter 
from io import BytesIO
from langchain_core.documents import Document
import fitz
import pandas as pd 
import numpy as np


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

# Atlas Vector Search Index 
VECTOR_SEARCH_INDEX_NAME = "vector_index"

app.config['VECTOR_SEARCH_INDEX_NAME'] = VECTOR_SEARCH_INDEX_NAME


# Load Speech-to-Text pipeline
try:
    from transformers import pipeline
    # Using a distilled, generally robust Whisper model for speed and good accent handling
    stt_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2")
    print("distill-whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading ASR model: {e}")
    stt_pipeline = None

# Create uploads directory
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['AUDIO_FOLDER'] = 'audio_responses'
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# --- Helper Functions ---

def embed_text(text: str, is_query: bool = False) -> list | None: # <--- ADDED `is_query` PARAMETER
    """
    Generates an embedding for the given text using the nomic-embed-text-v1 model.
    Applies the appropriate Nomic Embed task prefix ('search_query:' or 'search_document:').
    """
    if not embedding_model:
        print("Embedding model not loaded. Cannot embed text.")
        return None

    if not text or not text.strip():
        print("Empty text provided for embedding.")
        return None

    try:
        text = text.strip()
        if len(text) > 8000: # Arbitrary limit to prevent memory issues with long texts
            print(f"Text too long ({len(text)} chars), truncating.")
            text = text[:8000]
        
        # --- CRITICAL CHANGE: Apply Nomic Embed task prefix ---
        prefixed_text = f"search_query: {text}" if is_query else f"search_document: {text}"
        
        embedding = embedding_model.encode([prefixed_text], normalize_embeddings=True)
        
        if embedding is None or len(embedding) == 0:
            print("Embedding model returned None or empty result.")
            return None
            
        result = embedding[0].tolist()
        
        # Validate embedding dimensions
        if not isinstance(result, list) or len(result) == 0 or len(result) != EMBEDDING_DIMENSIONS:
            print(f"Invalid embedding format or dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(result) if isinstance(result, list) else 'N/A'}.")
            return None
            
        return result
        
    except Exception as e:
        print(f"Error during embedding for text (is_query={is_query}): {e}")
        return None

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


def chunk_document_langchain(document_path: str, file_extension: str):
    """
    Loads document and chunks it using RecursiveCharacterTextSplitter.
    Fixed to handle multiple file types properly.
    """
    documents = []
    try:
        print(f"Processing document: {document_path} with extension: {file_extension}")
        
        if file_extension.lower() == "pdf":
            documents_from_pdf = []
            try:
                pdf_document = fitz.open(document_path)
                print(f"Opened PDF with PyMuPDF. Total pages: {pdf_document.page_count}")

                for i in range(pdf_document.page_count):
                    page = pdf_document.load_page(i)
                    
                    page_content_parts = []
                    
                    # Store bounding boxes of already processed content (tables)
                    processed_bboxes = [] 
                    
                    print(f"--- Processing PDF Page {i+1} ---")

                    # --- Step 1: Attempt Table Extraction (Highest Priority) ---
                    try:
                        tables = page.find_tables()
                        if tables:
                            print(f"Page {i+1}: {len(tables.tables)} tables detected.")
                            for table_idx, table in enumerate(tables.tables):
                                bbox = tuple(table.bbox)
                                try:
                                    # Convert table to pandas DataFrame
                                    df = pd.DataFrame(table.extract())
                                    # Convert DataFrame to a string representation (e.g., Markdown or CSV-like)
                                    # Markdown is often good for readability in text-based chunks
                                    table_string = df.to_markdown(index=False) 
                                    
                                    if table_string.strip():
                                        page_content_parts.append({"type": "table", "content": table_string, "bbox": bbox})
                                        processed_bboxes.append(bbox)
                                        print(f"Page {i+1}, Table {table_idx}: Extracted table data.")
                                    else:
                                        print(f"Page {i+1}, Table {table_idx}: Table detected but extracted content was empty.")
                                except Exception as table_extract_e:
                                    print(f"Page {i+1}, Table {table_idx}: Error extracting table to DataFrame: {table_extract_e}")
                        else:
                            print(f"Page {i+1}: No tables detected.")
                    except Exception as table_detect_e:
                        print(f"Page {i+1}: Error detecting tables: {table_detect_e}")

                    # --- Step 2: Attempt structured block extraction (native text + image block OCR) ---
                    # Filter blocks to avoid re-processing areas covered by tables
                    page_blocks = page.get_text("blocks") 
                    if page_blocks:
                        # Filter blocks that overlap significantly with already extracted tables
                        filtered_blocks = []
                        for block in page_blocks:
                            block_bbox = fitz.Rect(block[:4]) # Create a fitz.Rect object for the block
                            is_overlapped = False
                            for p_bbox in processed_bboxes:
                                processed_rect = fitz.Rect(p_bbox)
                                if block_bbox.intersects(processed_rect) and block_bbox.inside(processed_rect):
                                    is_overlapped = True
                                    break
                            if not is_overlapped:
                                filtered_blocks.append(block)
                        
                        if len(filtered_blocks) < len(page_blocks):
                            print(f"Page {i+1}: Skipped {len(page_blocks) - len(filtered_blocks)} blocks due to table overlap.")

                        if not filtered_blocks:
                             print(f"Page {i+1}: No non-overlapped blocks to process after table extraction.")
                             
                        for block_idx, block in enumerate(filtered_blocks):
                            x0, y0, x1, y1, content, block_no, block_type = block
                            
                            if block_type == 0:  # This is a text block
                                text = content.strip()
                                if text:
                                    page_content_parts.append({"type": "text", "content": text, "bbox": (x0, y0, x1, y1)})
                                    # print(f"Page {i+1}, Block {block_idx}: Native text found, length: {len(text)} characters.")
                            elif block_type == 1: # This is an image block
                                # print(f"Page {i+1}, Block {block_idx}: Image block detected. Attempting OCR.")
                                try:
                                    pix = page.get_pixmap(clip=(x0, y0, x1, y1))
                                    img_data = pix.pil_tobytes(format="PNG")
                                    image_to_ocr = Image.open(io.BytesIO(img_data))
                                    
                                    # Apply same preprocessing as for standalone JPGs
                                    image_to_ocr = image_to_ocr.convert('L') # Grayscale
                                    enhancer = ImageEnhance.Contrast(image_to_ocr)
                                    image_to_ocr = enhancer.enhance(1.5) # Increase contrast
                                    image_to_ocr = image_to_ocr.filter(ImageFilter.SHARPEN) # Sharpen
                                    
                                    extracted_text = pytesseract.image_to_string(image_to_ocr, lang='eng')
                                    
                                    if extracted_text.strip():
                                        page_content_parts.append({"type": "ocr", "content": extracted_text, "bbox": (x0, y0, x1, y1)})
                                        # print(f"Page {i+1}, Block {block_idx}: OCR successfully extracted text from image block.")
                                except Exception as img_ocr_e:
                                    print(f"Page {i+1}, Block {block_idx}: Error during OCR of image block: {img_ocr_e}")
                    else:
                        print(f"Page {i+1}: No PyMuPDF blocks detected for standard text/image processing.")
                    
                    # Sort all content parts (tables, text, OCR) by their vertical position to maintain reading order
                    page_content_parts.sort(key=lambda x: x["bbox"][1]) 
                    combined_page_text = "\n\n".join([part["content"] for part in page_content_parts if part["content"]])
                    
                    # --- Step 3: Fallback to full-page OCR if no meaningful content found from blocks (including tables) ---
                    # Keep the original filtering logic here, as per your preference.
                    if not (combined_page_text.strip() and any(char.isalnum() for char in combined_page_text)):
                        print(f"Page {i+1}: No usable content from structured extraction (tables/blocks). Attempting full-page OCR.")
                        try:
                            # Render the entire page as an image
                            pix = page.get_pixmap() # Get pixmap of entire page
                            img_data = pix.pil_tobytes(format="PNG")
                            full_page_image = Image.open(io.BytesIO(img_data))

                            # Apply preprocessing to the full-page image
                            full_page_image = full_page_image.convert('L')
                            enhancer = ImageEnhance.Contrast(full_page_image)
                            full_page_image = enhancer.enhance(1.5)
                            full_page_image = full_page_image.filter(ImageFilter.SHARPEN)
                            if full_page_image.width < 1000 and full_page_image.height < 1000: # Example threshold
                                full_page_image = full_page_image.resize((full_page_image.width * 2, full_page_image.height * 2), Image.LANCZOS)
                            
                            full_page_ocr_text = pytesseract.image_to_string(full_page_image, lang='eng')
                            
                            if full_page_ocr_text.strip() and any(char.isalnum() for char in full_page_ocr_text):
                                combined_page_text = full_page_ocr_text
                                print(f"Page {i+1}: Full-page OCR successfully extracted text.")
                            else:
                                print(f"Page {i+1}: Full-page OCR found no usable text.")
                        except Exception as full_ocr_e:
                            print(f"Page {i+1}: Error during full-page OCR: {full_ocr_e}")

                    # --- Step 4: Add content to documents if usable ---
                    print(f"Page {i+1}: Final combined content length for page: {len(combined_page_text)} characters.")
                    
                    if combined_page_text.strip() and any(char.isalnum() for char in combined_page_text):
                        documents_from_pdf.append(Document(page_content=combined_page_text, metadata={"source": document_path, "page": i+1, "file_type": "pdf_mixed_content"}))
                        print(f"Page {i+1}: Document added to list.")
                    else:
                        print(f"Page {i+1}: No usable content (tables, blocks, or full-page OCR) found, document NOT added.")
                
                documents = documents_from_pdf 
                print(f"Total {len(documents)} usable page documents extracted from PDF.")
                
            except pytesseract.TesseractNotFoundError:
                print("Tesseract is not installed or not in your PATH. Please install it to enable OCR for PDFs.")
                raise # Re-raise for error handling in upload_document
            except Exception as pdf_e:
                print(f"Error processing PDF with PyMuPDF: {pdf_e}")
                return [] # Return empty if PDF processing fails # Return empty if PDF processing fails
                
        elif file_extension.lower() == "txt":
            loader = TextLoader(document_path, encoding='utf-8')
            documents = loader.load()
            print(f"Loaded {len(documents)} text documents")
            
        elif file_extension.lower() in ["docx", "doc"]:
            loader = Docx2txtLoader(document_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from Word file")

        elif file_extension.lower() in ["jpg", "jpeg", "png"]:
            print(f"Attempting OCR on image file: {document_path}")
            try:
                image = Image.open(document_path)

                # --- Image Preprocessing Steps (Highly Recommended) ---
                image = image.convert('L') # 'L' for luminance (grayscale)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5) # Increase contrast by 50%
                image = image.filter(ImageFilter.SHARPEN)

                if image.width < 1000 and image.height < 1000: # Example threshold
                     image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

                extracted_text = pytesseract.image_to_string(image, lang='eng')

                if extracted_text.strip():
                    documents = [Document(page_content=extracted_text, metadata={"source": document_path, "file_type": "image_ocr"})]
                    print(f"Successfully extracted text from image: {len(extracted_text)} characters.")
                else:
                    print(f"No text found in image {document_path} after OCR. This might be due to low quality or no text in image.")
                    return []
            except pytesseract.TesseractNotFoundError:
                print("Tesseract is not installed or not in your PATH. Please install it.")
                raise 
            except Exception as e:
                print(f"Error during OCR processing for {document_path}: {e}")
                return []
            
    
            
             
            print(f"Total {len(documents)} usable documents (from native text or OCR) extracted from PDF.")

            
        elif file_extension.lower() == "txt":
            loader = TextLoader(document_path, encoding='utf-8')
            documents = loader.load()
            print(f"Loaded {len(documents)} text documents")
            
        elif file_extension.lower() in ["docx", "doc"]:
            # Use Docx2txtLoader for Word documents
            loader = Docx2txtLoader(document_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents from Word file")

        elif file_extension.lower() in ["jpg", "jpeg", "png"]:
            print(f"Attempting OCR on image file: {document_path}")
            try:
                image = Image.open(document_path)

                # --- Image Preprocessing Steps (Highly Recommended) ---
                # 1. Convert to Grayscale
                image = image.convert('L') # 'L' for luminance (grayscale)

                # 2. Enhance Contrast (optional, but often helps)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5) # Increase contrast by 50%

                # 3. Apply a Sharpen filter (optional)
                image = image.filter(ImageFilter.SHARPEN)

                # 4. Upscale (if image is low resolution, e.g., < 300 DPI)
                if image.width < 1000 and image.height < 1000: # Example threshold
                     image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

                extracted_text = pytesseract.image_to_string(image,lang = 'eng')

                if extracted_text.strip():
                    documents = [Document(page_content=extracted_text, metadata={"source": document_path, "file_type": "image_ocr"})]
                    print(f"Successfully extracted text from image: {len(extracted_text)} characters.")
                else:
                    print(f"No text found in image {document_path} after OCR. This might be due to low quality or no text in image.")
                    return []
            except pytesseract.TesseractNotFoundError:
                print("Tesseract is not installed or not in your PATH. Please install it.")
                raise # Re-raise to be caught by the outer try-except in upload_document
            except Exception as e:
                print(f"Error during OCR processing for {document_path}: {e}")
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
        chunk_size=512,
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
            if chunk_text and any(char.isalnum() for char in chunk_text):  # Only add non-empty chunks
                chunk_texts.append(chunk_text)
                print(f"Chunk {i+1}: {len(chunk_text)} characters")
        
        print(f"Final chunk count after filtering: {len(chunk_texts)}")
        return chunk_texts
        
    except Exception as e:
        print(f"Error during text splitting: {e}")
        return []
    

def perform_vector_search(query_embedding: list, selected_doc_ids: list, limit: int = 10) -> list:
    if not query_embedding:
        print("Error: Query embedding is empty or None. Cannot perform vector search.")
        return []

    # --- IMPORTANT: Get user_email from Flask session inside this function ---
    user_email = session.get("email") 
    if not user_email:
        print("Error: User email not found in session for vector search filtering. Returning no results.")
        return []

    # Build the filter criteria for $vectorSearch
    # This filter directly uses the fields you defined as "filter" type in your Atlas index
    search_filter = {"user_email": user_email}
    if selected_doc_ids:
        print(f"Filtering vector search by {len(selected_doc_ids)} selected documents for user {user_email}.")
        search_filter["doc_id"] = {"$in": selected_doc_ids}
    else:
        print(f"No specific documents selected. Searching all available chunks for user {user_email}.")

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_SEARCH_INDEX_NAME, # Your Atlas Search Index name
                "path": "embedding",                # The field containing the vector in your document_chunks
                "queryVector": query_embedding,     # The embedding of the user's query
                "numCandidates": 100,               # Number of approximate nearest neighbors to consider
                "limit": limit,                     # Number of top results to return
                "filter": search_filter             # Apply your filters here
            }
        },
        {
            # Project only the necessary fields to reduce data transfer
            "$project": {
                "text": 1,
                "doc_id": 1,
                "_id": 0,                           # Exclude the default MongoDB _id
                "score": {"$meta": "vectorSearchScore"} # Get the relevance score (changed from rrf_score)
            }
        }
    ]

    try:
        # Execute the aggregation pipeline on the document_chunks collection
        results = list(db.document_chunks.aggregate(pipeline))
        print(f"Vector search returned {len(results)} relevant chunks.")
        return results
    except Exception as e:
        print(f"Error during MongoDB Atlas Vector Search: {e}")
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

# --- CHAT INTERACTION ---
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
    
    ai21_messages = []

    system_prompt_general = (
        '''You are a helpful, respectful, and honest assistant
        Always be safe, and do not generate content that is harmful, hateful, 
        illegal, or violent. 
        If you are unsure how to respond to a query, or if the information 
        you are asked to recall is not within the scope of a general assistant, 
        you can politely state that you cannot fulfill the request or that you 
        do not have that specific information.'''
    )
    
    bot_reply = "Sorry, I couldn't generate a response due to an internal error."

    try:
        if mode == "document":
            selected_doc_ids = chat.get("selected_documents", [])
            if not selected_doc_ids:
                return jsonify({"reply": "Please select documents to use RAG mode."}), 400

            query_embedding = embed_text(user_message, is_query=True)
            if query_embedding is None:
                return jsonify({"reply": "Failed to create embedding for your query."}), 500

            # Use the corrected hybrid search
            start_search_time = time.time()
            top_chunks = perform_vector_search(
                query_embedding=query_embedding,
                selected_doc_ids=selected_doc_ids,
                limit=5
            )
            search_time = time.time() - start_search_time
            print(f"Vector search took {search_time:.3f} seconds")

            if not top_chunks:
                return jsonify({"reply": "No relevant information found in the selected documents."}), 200
            
            RELEVANCE_THRESHOLD = 0.55 

            relevant_chunks = [chunk for chunk in top_chunks if chunk.get('score', 0) >= RELEVANCE_THRESHOLD]
            
            if not relevant_chunks:
                print(f"No sufficiently relevant information found for query '{user_message}' (score < {RELEVANCE_THRESHOLD}).")
                return jsonify({"reply": "I'm sorry, I couldn't find information relevant to your request within the selected documents. Please try rephrasing or selecting different documents."}), 200

            # Fetch document names for context
            doc_names_map = {}
            if selected_doc_ids:
                doc_metadata = list(db.document_metadata.find({"doc_id": {"$in": selected_doc_ids}}))
                for doc in doc_metadata:
                    doc_names_map[doc["doc_id"]] = doc.get("filename", "Unknown Document")

            # Create context from search results
            context_parts = []
            seen_doc_ids = set() # To ensure document name is only added once per doc
            for chunk in top_chunks:
                # Get document name, default to chunk's doc_id if not found
                doc_name = doc_names_map.get(chunk['doc_id'], chunk['doc_id'])
                
                # Add document name header only once per document
                if chunk['doc_id'] not in seen_doc_ids:
                    context_parts.append(f"--- Document: {doc_name} ---")
                    seen_doc_ids.add(chunk['doc_id'])

                context_parts.append(f"[Score: {chunk.get('score', 0):.4f}] {chunk['text']}") 
            
            context = "\n\n".join(context_parts)

            # RAG-specific system prompt
            system_prompt_rag = (
            '''You are a helpful, factual, and concise assistant. Your primary goal is to answer user questions truthfully and directly **based solely on the provided context or documents**.
            Here are the strict guidelines you must follow:
            1.**Prioritize Context:** Always use the information provided in the [DOCUMENTS] section to formulate your answers.
            2.**Do Not Hallucinate:** If the answer to the question **cannot be found within the provided [DOCUMENTS]**, state clearly and politely that the information is not available in the given context. Do not make up answers or draw on outside knowledge.
            3.**Answer Directly:** Provide concise and direct answers without unnecessary conversational filler. For direct facts, extract them as-is.
            4.**Synthesize if needed:** If the user asks for a summary or requires combining information from multiple sentences within the [DOCUMENTS], you may do so, but *only* using the information present.
            5.**Maintain Neutral Tone:** Your responses should be neutral and objective.
            6.**Inferences": if the user asks anything refering the name of the documents you should infer by the name 
            of the document that what document is being talked about and if user asks anything that is not directly mentioned 
            in the [DOCUMENTS] or context but can be inferred using basic human logic , then u should infer it but
            u must confirm with the user if your inference makes sense before actually displaying your inference

[DOCUMENTS]
''' + context + '''
[/DOCUMENTS]'''
            )
            ai21_messages.insert(0, ChatMessage(content=system_prompt_rag, role="system"))
            ai21_messages.append(ChatMessage(content=user_message, role="user"))
        else:
            ai21_messages = _convert_history_for_ai21(history)
            ai21_messages.append(ChatMessage(content=user_message, role="user"))
            ai21_messages.insert(0, ChatMessage(content=system_prompt_general, role="system"))

        # Generate response
        response = ai21_client.chat.completions.create(
            messages=ai21_messages,
            model="jamba-mini"
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
    Handles document upload, text extraction, chunking, embedding,
    and storage in the database.
    """
    try:
        # Validate request
        if 'file' not in request.files:
            print("Error: No file part in request.")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        if file.filename == '':
            print("Error: No selected file.")
            return jsonify({"error": "No selected file"}), 400

        chat_id = request.form.get("chat_id")
        if not chat_id:
            print("Error: chat_id is required for document upload.")
            return jsonify({"error": "chat_id is required"}), 400

        email = session.get("email")
        if not email:
            print("Error: Unauthorized access - user email not found in session.")
            return jsonify({"error": "Unauthorized"}), 401

        # Validate file type and filename
        filename = secure_filename(file.filename)
        if not filename:
            print("Error: Invalid filename provided.")
            return jsonify({"error": "Invalid filename"}), 400
            
        file_extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        allowed_extensions = ['pdf', 'txt', 'docx', 'doc', 'jpg', 'jpeg', 'png']
        
        if file_extension not in allowed_extensions:
            print(f"Error: Unsupported file type '{file_extension}'. Allowed: {', '.join(allowed_extensions)}")
            return jsonify({"error": f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"}), 400

        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size == 0:
            print("Error: Uploaded file is empty.")
            return jsonify({"error": "File is empty"}), 400

        print(f"Starting to process file: {filename} (Size: {file_size} bytes, Ext: {file_extension}) for chat: {chat_id}, user: {email}")

        # Save file temporarily
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            print(f"Temporary file saved to: {filepath}")
        except Exception as e:
            print(f"Error saving file '{filename}' temporarily: {e}")
            return jsonify({"error": "Failed to save uploaded file temporarily"}), 500

        # Process document (extract and chunk text)
        final_chunks_text_content = []
        try:
            final_chunks_text_content = chunk_document_langchain(filepath, file_extension)
            print(f"Successfully extracted {len(final_chunks_text_content)} text chunks from '{filename}'.")
        except Exception as e:
            print(f"Error processing document '{filename}' for chunking: {e}")
            return jsonify({"error": f"Failed to process document content: {str(e)}"}), 500
        finally:
            # Always clean up the temporary file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Temporary file removed: {filepath}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {filepath} after processing: {e}")

        if not final_chunks_text_content:
            print(f"Error: No text content could be extracted from '{filename}' or all chunks were empty.")
            return jsonify({"error": "No text content could be extracted from the document"}), 400

        # Create embeddings and prepare chunks for storage
        doc_id = str(uuid.uuid4())
        stored_chunks_count = 0
        chunks_to_insert = []
        failed_embeddings = 0
        
        for i, chunk_text_content in enumerate(final_chunks_text_content):
            if not chunk_text_content or not chunk_text_content.strip(): # Skip empty or whitespace-only chunks
                continue
                
           
            embedding = embed_text(chunk_text_content, is_query=False) # <--- Pass is_query=False
            
            if embedding is not None:
                chunks_to_insert.append({
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "user_email": email,
                    "text": chunk_text_content,
                    "embedding": embedding,
                    "created_at": utcnow() # Ensure utcnow() is defined and returns datetime object
                })
                stored_chunks_count += 1
            else:
                failed_embeddings += 1
                print(f"Failed to create embedding for chunk {i+1} of doc '{filename}'.")

        if stored_chunks_count == 0:
            print(f"Error: No valid chunks could be processed and embedded for doc '{filename}'.")
            return jsonify({"error": "No chunks could be processed and embedded"}), 500

        if failed_embeddings > 0:
            print(f"Warning: {failed_embeddings} chunks failed to embed for doc '{filename}'.")

        # Store chunks in database
        try:
            result = db.document_chunks.insert_many(chunks_to_insert)
            print(f"Successfully inserted {len(result.inserted_ids)} chunks into 'document_chunks' collection for doc_id: {doc_id}")
        except Exception as e:
            print(f"Error inserting chunks into database for doc_id {doc_id}: {e}")
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
            print(f"Error storing document metadata for doc_id {doc_id}: {e}")
            # Try to clean up chunks if metadata fails to prevent orphaned chunks
            try:
                db.document_chunks.delete_many({"doc_id": doc_id})
                print(f"Cleaned up chunks for doc_id {doc_id} due to metadata storage failure.")
            except Exception as cleanup_e:
                print(f"Error during chunk cleanup for doc_id {doc_id}: {cleanup_e}")
            return jsonify({"error": "Failed to store document metadata"}), 500

        # Add document to chat's selected documents
        try:
            result = db.conversations.update_one(
                {"chat_id": chat_id, "user_email": email},
                {
                    "$addToSet": {"selected_documents": doc_id}, # Ensure doc_id is added
                    "$set": {"updated_at": utcnow()}
                }
            )
            if result.matched_count == 0:
                print(f"Warning: Chat {chat_id} not found when attempting to add document {doc_id}. It might be a new chat or an issue with chat_id/user_email.")
            else:
                print(f"Document {doc_id} added to chat {chat_id} selected documents.")
        except Exception as e:
            print(f"Error adding document {doc_id} to chat {chat_id}'s selected_documents: {e}")
            # Don't fail the entire operation for this, as the document itself is stored

        print(f"Document '{filename}' (doc_id: {doc_id}) upload process completed.")
        return jsonify({
            "status": "uploaded",
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": stored_chunks_count,
            "failed_embeddings": failed_embeddings
        })

    except Exception as e:
        print(f"Unexpected top-level error in upload_document: {e}")
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

@app.route("/tts", methods=["POST"])
def text_to_speech():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided for TTS"}), 400

    try:
        # Create a gTTS object
        tts = gTTS(text=text, lang='en', tld='com', slow=False) # 'en' for English, 'tld' for accent like 'co.in' for Indian accent

        # Save audio to a BytesIO object (in-memory) instead of a file
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0) # Rewind to the beginning of the stream

        # Send the audio data back as a file
        return send_file(audio_stream, mimetype="audio/mpeg", as_attachment=False, download_name="response.mp3")

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        return jsonify({"error": f"Failed to generate speech: {e}"}), 500
    
    
# --- Run the app ---
if __name__ == "__main__":
    # Ensure MongoDB and AI21 clients are initialized
    if db is None or ai21_client is None:
        print("Application cannot start due to initialization errors. Please check previous error messages.")
        exit(1) # Exit if essential services are not available

  
    app.run(debug=True, use_reloader=False) 
