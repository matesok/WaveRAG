# WaveRAG

WaveRAG is a Retrieval-Augmented Generation (RAG) system that leverages document embeddings and a language model to answer questions based on your own documents. It uses ChromaDB as a vector store, LangChain for orchestration, and FastAPI for serving a web API and static frontend.

## Features
- Ingests and indexes Markdown and PDF documents (from a zip archive)
- Stores document embeddings in ChromaDB
- Uses Ollama for local LLM and embedding models
- Provides a FastAPI-based backend with a chat endpoint
- Serves a static HTML frontend
- Dockerized for easy deployment

## Requirements
- Python 3.11+
- pip
- Docker (optional, for containerized deployment)
- Ollama server running locally (for LLM and embeddings)

## Installation

### Local (development)
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd WaveRAG
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running locally (default: http://localhost:11434).
4. Start the FastAPI server:
   ```sh
   uvicorn main:app --reload
   ```
5. Open [http://localhost:8000](http://localhost:8000) in your browser.

### Docker Compose
You can use Docker Compose to run both the Ollama server and the WaveRAG backend together:

1. Start the services:
   ```sh
   docker-compose up
   ```
   This will launch two containers:
   - `ollama`: LLM and embedding server (pulls required models on first run)
   - `rag`: WaveRAG backend (FastAPI)

2. Access the app at [http://localhost:8000](http://localhost:8000)

**Note:**
- The first run may take a while as Ollama downloads the models (`llama3.2`, `nomic-embed-text`).
- Documents should be placed in the `documents/` directory as a zip file (`sdlc_doc.zip`).
- Data persists in Docker volumes (`ollama_data`, `chroma_data`).

## Project Structure
- `main.py` – FastAPI app, API endpoints, static file serving
- `rag/`
  - `chain.py` – RAG pipeline, prompt construction, LLM streaming
  - `db.py` – ChromaDB integration, embedding setup
  - `ingest_docs.py` – Document loading, splitting, and ingestion
- `documents/` – Place your `sdlc_doc.zip` (containing Markdown/PDFs) here
- `chroma_db/` – Persistent vector store (auto-created)
- `static/` – Frontend files (e.g., `index.html`)
- `requirements.txt` – Python dependencies
- `Dockerfile` – Container build instructions

## Usage
- On startup, documents from `documents/sdlc_doc.zip` are extracted, split, embedded, and stored in ChromaDB.
- The `/chat` endpoint accepts POST requests with a question and streams the answer from the LLM, grounded in your documents.
- The root endpoint `/` serves the static frontend.

### Example API Request
```http
POST /chat
Content-Type: application/json
{
  "question": "What is the SDLC process?"
}
```

## Environment Variables
- `OLLAMA_BASE_URL` – URL for the Ollama server (default: `http://localhost:11434`)

## License
MIT License (see LICENSE file for details)
