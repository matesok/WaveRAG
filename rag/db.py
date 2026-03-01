import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "my_docs"


def load_documents_to_db(documents_chunks):
    db = Chroma.from_documents(
        documents_chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    return db


def get_db():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    if db._collection.count() == 0:
        return None

    return db
