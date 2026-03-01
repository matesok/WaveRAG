import os
from typing import Generator

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rag.db import get_db
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(model="llama3.2", base_url=OLLAMA_URL)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""


def query_rag(question: str) -> Generator:
    db = get_db()
    results = db.similarity_search_with_relevance_scores(question, k=3)
    print(results)
    context_text = "\n\n----\n\n".join([doc.page_content for doc, score in results])

    prompt = (ChatPromptTemplate.
    from_template(PROMPT_TEMPLATE)
    .format(
        context=context_text,
        question=question
    ))

    for chunk in llm.stream(prompt):
        yield chunk.content

    sources = "\n".join([
        f"- {os.path.basename(doc.metadata['source'])}"
        for doc, score in results
        if score >= 0.5
    ])

    if sources:
        yield f"\n\n---\n\n**Sources:**\n{sources}"
