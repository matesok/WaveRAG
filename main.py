import logging
import sys

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rag.chain import query_rag
from rag.db import get_db
from rag.ingest_docs import load_and_store_documents

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


@app.on_event("startup")
async def startup():
    db = get_db()

    if db is None:
        logging.info("DB empty, loading documents...")
        load_and_store_documents()
    else:
        logging.info(f"DB already has {db._collection.count()} chunks, skipping ingestion")


configure_logging()


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(request: QueryRequest):
    return StreamingResponse(query_rag(request.question), media_type='text/plain')
