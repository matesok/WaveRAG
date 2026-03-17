import os
import time
from typing import Generator

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rag.db import get_db

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(model="llama3.2", base_url=OLLAMA_URL)

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
"""


def query_rag(question: str) -> Generator:
    t9 = time.time()
    db = get_db()
    print(f"Time after start DB {time.time()-t9:.2f}s")
    t0 = time.time()
    results = db.max_marginal_relevance_search(question,  k=2, fetch_k=8)
    print(f"Retrieval (embedding + search): {time.time()-t0:.2f}s")
    
    print(f"QUESTION {question} ALL RESULTS:")
    for doc in results:
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content[:100])
        print("---")

    if not results:
        yield "No results found."
        return

    context_text = "\n\n----\n\n".join([doc.page_content for doc in results])
    t1 = time.time()
    prompt = (ChatPromptTemplate
        .from_template(PROMPT_TEMPLATE)
        .format(context=context_text, question=question))
    
    for chunk in llm.stream(prompt):
        yield chunk.content
    print(f"LLM generation: {time.time()-t1:.2f}s")
    sources = "\n".join([
        f"- {os.path.basename(doc.metadata['source'])}"
        for doc in results
    ])

    if sources:
        yield f"\n\n---\n\n**Sources:**\n{sources}"
