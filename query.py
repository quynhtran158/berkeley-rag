"""
query.py — Retrieval and grounded generation pipeline.

Given a user question:
1. Embed the query using all-MiniLM-L6-v2
2. Retrieve the top-5 most semantically similar chunks from ChromaDB
3. Construct a grounded prompt that restricts the LLM to retrieved context only
4. Generate a response via Groq's llama-3.3-70b-versatile
5. Return the answer and source citations
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from embed import load_retriever

load_dotenv()

TOP_K = 5
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the Unofficial Berkeley CS Guide — a knowledgeable assistant that answers questions about UC Berkeley CS courses, professors, campus life, housing, and student resources.

CRITICAL RULES:
1. Answer ONLY using information explicitly stated in the CONTEXT DOCUMENTS provided below.
2. Do NOT use your general training knowledge to fill in gaps, make inferences, or add information not present in the documents.
3. If the provided documents do not contain enough information to answer the question, respond with: "I don't have enough information in my documents to answer that question reliably."
4. You MUST cite your sources. At the end of every response, list the source documents you drew from, formatted as "Sources: [filename1, filename2]".
5. If a question asks you to compare two things (e.g., two professors), only state comparisons that are explicitly made in the documents — do not infer unstated comparisons.
"""

_model = None
_collection = None


def _load_components():
    global _model, _collection
    if _model is None or _collection is None:
        _model, _collection = load_retriever()


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and retrieve the top_k most relevant chunks.
    Returns a list of dicts: {text, source, distance}.
    """
    _load_components()
    query_embedding = _model.encode([query], convert_to_list=True)
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "distance": dist,
        })
    return chunks


def build_context(chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Returns (context_string, list_of_unique_sources).
    """
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Document {i} — Source: {chunk['source']}]\n{chunk['text']}"
        )
        if chunk["source"] not in sources:
            sources.append(chunk["source"])
    return "\n\n---\n\n".join(context_parts), sources


def generate(query: str, chunks: list[dict]) -> str:
    """
    Generate a grounded answer using Groq's LLM.
    The system prompt enforces answering only from retrieved context.
    """
    context, _ = build_context(chunks)

    user_message = f"""CONTEXT DOCUMENTS:
{context}

---

QUESTION: {query}

Answer the question using ONLY the information in the context documents above. Cite your sources at the end."""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def ask(question: str) -> dict:
    """
    End-to-end RAG pipeline: retrieve + generate.
    Returns a dict with 'answer', 'sources', and 'retrieved_chunks'.
    """
    chunks = retrieve(question)
    answer = generate(question, chunks)
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
    }


if __name__ == "__main__":
    test_questions = [
        "What is the difference between taking CS61B with Hilfinger vs. Hug?",
        "Which dining hall has the shortest wait times?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print(f"\nTop retrieved chunk (distance {result['retrieved_chunks'][0]['distance']:.4f}):")
        print(result["retrieved_chunks"][0]["text"][:300])
