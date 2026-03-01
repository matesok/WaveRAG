import os
import tempfile
import zipfile
import logging
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

from rag.db import load_documents_to_db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(BASE_DIR, "documents", "sdlc_doc.zip")

def load_and_store_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        markdown_loader = DirectoryLoader(
            tmpdir,
            glob="**/*.md",
            loader_cls=TextLoader
        )

        markdown_docs = markdown_loader.load()
        logging.info('Loaded markdown documents: %s', len(markdown_docs))

        pdf_loader = PyPDFDirectoryLoader(tmpdir)
        pdf_docs = pdf_loader.load()
        logging.info('Loaded pdf documents: %s', len(pdf_docs))

    markdown_chunks = MarkdownTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
    ).split_documents(markdown_docs)
    logging.info('Loaded markdown splits: %s', len(markdown_chunks))

    pdf_chunks = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
    ).split_documents(pdf_docs)
    logging.info('Loaded pdf splits: %s', len(pdf_chunks))

    total_chunks = markdown_chunks + pdf_chunks
    print("Adding documents to vector store...")
    load_documents_to_db(total_chunks)
    print("Done!")
