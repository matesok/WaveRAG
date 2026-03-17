import os
import logging
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "my_docs"

_cache = {"embeddings": None}

def get_embeddings():
    if _cache["embeddings"] is None:
        _cache["embeddings"] = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_URL, num_ctx=512)
    return _cache["embeddings"]

def load_documents_to_db(documents_chunks):
    BATCH_SIZE = 50
    db = None
    
    for i in range(0, len(documents_chunks), BATCH_SIZE):
        batch = documents_chunks[i:i+BATCH_SIZE]
        logging.info(f"Indexing batch {i//BATCH_SIZE + 1}/{(len(documents_chunks)-1)//BATCH_SIZE + 1} ({len(batch)} chunks)")
        
        if db is None:
            db = Chroma.from_documents(
                batch,
                get_embeddings(),
                persist_directory=CHROMA_PATH,
                collection_name=COLLECTION_NAME,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            db.add_documents(batch)
    
    return db


def get_db():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME
    )

    if db._collection.count() == 0:
        return None

    return db
