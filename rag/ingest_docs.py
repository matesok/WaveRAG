import os
import re
import tempfile
import zipfile
import logging
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

from rag.db import load_documents_to_db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(BASE_DIR, "documents", "sdlc_doc.zip")

def clean_document(doc):
    content = re.sub(r'^---.*?---\s*', '', doc.page_content, flags=re.DOTALL)
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    doc.page_content = content
    return doc

def load_and_store_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        markdown_docs = DirectoryLoader(tmpdir, glob="**/*.md", loader_cls=TextLoader,loader_kwargs={"encoding": "utf-8"}).load()
        markdown_docs = [clean_document(doc) for doc in markdown_docs]
        logging.info('Loaded markdown documents: %s', len(markdown_docs))
        for doc in markdown_docs:
            print(doc.metadata['source'])
        pdf_docs = PyPDFDirectoryLoader(tmpdir).load()
        logging.info('Loaded pdf documents: %s', len(pdf_docs))

        markdown_chunks = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(markdown_docs)
        pdf_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(pdf_docs)

    total_chunks = markdown_chunks + pdf_chunks
    logging.info('Total chunks: %s', len(total_chunks))
    load_documents_to_db(total_chunks)