# Retrieval-Augmented Generation (RAG) Chatbot

---

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot with document management capabilities and integrated speech-to-text (STT) and text-to-speech (TTS) functionalities. It allows users to upload their own documents, which are then processed, chunked, and embedded to serve as a knowledge base for the chatbot. The AI model leverages this context to provide accurate and relevant answers, as well as provide a global mode for chat in general.

---

## Features

* **RAG-powered Chatbot**: Answers user queries by retrieving information directly from uploaded documents.
* **Document Upload**: Supports various document types (PDF, TXT, DOCX, DOC, JPG, JPEG, PNG) for building a custom knowledge base.
* **Automatic Chunking & Embedding**: Uploaded documents are automatically broken down into smaller chunks and converted into vector embeddings for efficient similarity search.
* **Document Management**:
    * Rename uploaded documents.
    * Delete documents and their associated data.
    * Select/deselect specific documents for a given chat session to control the RAG context.
* **Conversational History**: Maintains chat history for ongoing conversations.
* **Auto-Generated Chat Names**: Automatically names new chat sessions based on the initial conversation.
* **Speech-to-Text (STT)**: Transcribes spoken audio input into text for chat queries.
* **Text-to-Speech (TTS)**: Converts the chatbot's text responses into natural-sounding speech.
* **User Authentication**: A user login system to manage individual user documents and chats.

---

## Technologies Used

* **Backend**: Flask (Python Web Framework)
* **Database**: MongoDB (for storing chat history, document metadata, and document chunks)
* **Vector Embeddings**:nomic-ai/nomic-embed-text-v1 , view more models on https://huggingface.co/spaces/mteb/leaderboard
* **Large Language Model (LLM)**: AI21 Jamba-Mini (for generating chat responses)
* **Speech-to-Text**: Hugging Face Transformers `pipeline` (distil-whisper/distil-large-v2)
* **Text-to-Speech**: gTTS (Google Text-to-Speech)
* **Document Processing**: Pymupdf , langchain text splitters and document laoders
* **Frontend**: HTML,CSS,JS

---

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.8+
* MongoDB instance (local or cloud-hosted)
* AI21 API Key
* (If applicable, any specific model weights or external dependencies for STT/Embeddings)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TheDevilspear/RAGbasedbot.git]
    cd RAGbasedbot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing all necessary libraries like Flask, pymongo, ai21, transformers, gTTS, langchain, python-dotenv, python-multipart, etc.)*

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory of your project and add the following:
    ```
    MONGO_URI="your_mongodb_connection_string"
    AI21_API_KEY="your_ai21_api_key"
    UPLOAD_FOLDER="./uploads" # Ensure this directory exists or create it
    SECRET_KEY="a_very_secret_key_for_flask_sessions"
    # Add any other API keys or configuration settings
    ```

### Running the Application

1.  **Start MongoDB:**
    Ensure your MongoDB instance is running and accessible via the `MONGO_URI` provided in your `.env` file.

2.  **Run the Flask application:**
    ```bash
    python app.py  # Or whatever your main Flask file is named
    ```
    The application will typically run on `http://127.0.0.1:5000`.

---

## API Endpoints

This section outlines the primary API endpoints exposed by the Flask backend.

* `POST /chat`: Main endpoint for sending user messages and getting chatbot responses.
    * **Request Body**: `{ "chat_id": "...", "user_message": "...", "history": [], "selected_doc_ids": [] }`
    * **Response**: `{ "reply": "...", "chat_id": "..." }`
* `POST /upload_document`: Uploads a document, processes it, and stores its chunks and metadata.
    * **Request Body**: `multipart/form-data` with `file` and `chat_id`.
    * **Response**: `{ "status": "uploaded", "doc_id": "...", "filename": "...", "chunks_count": ..., "failed_embeddings": ... }`
* `POST /rename_document`: Renames an uploaded document.
    * **Request Body**: `{ "doc_id": "...", "new_name": "..." }`
    * **Response**: `{ "status": "renamed" }`
* `POST /delete_document`: Deletes a document and its associated data.
    * **Request Body**: `{ "doc_id": "..." }`
    * **Response**: `{ "status": "deleted" }`
* `POST /toggle_document`: Adds or removes a document from the current chat's selected documents.
    * **Request Body**: `{ "chat_id": "...", "doc_id": "...", "action": "select" | "deselect" }`
    * **Response**: `{ "status": "updated" }`
* `POST /speech_input`: Accepts audio input for transcription.
    * **Request Body**: `multipart/form-data` with `audio` file.
    * **Response**: `{ "transcribed_text": "..." }`
* `POST /tts`: Converts text to speech.
    * **Request Body**: `{ "text": "..." }`
    * **Response**: `audio/mpeg` file stream.
* `GET /get_documents`: (Assumed, for listing user's uploaded documents)
* `GET /get_chats`: (Assumed, for listing user's chat sessions)

---

## Project Structure (Conceptual)
