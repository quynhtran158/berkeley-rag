"""
embed.py — Embedding pipeline and ChromaDB vector store setup.

Embeds all document chunks using all-MiniLM-L6-v2 and stores them
in a local ChromaDB collection with source metadata.
"""

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import build_chunks

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "berkeley_cs_guide"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def get_collection(reset: bool = False) -> chromadb.Collection:
    """
    Initialize (or load) the ChromaDB persistent collection.
    If reset=True, delete and recreate the collection.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(chunks: list[dict], collection: chromadb.Collection,
                    model: SentenceTransformer) -> None:
    """
    Embed each chunk and upsert into ChromaDB.
    Each document in the collection stores:
      - id: unique string
      - embedding: 384-dim float vector
      - document: the chunk text (for retrieval display)
      - metadata: source filename + chunk index
    """
    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}__chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]}
                 for c in chunks]

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME}...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_list=True)

    # Upsert in batches of 100 to avoid memory issues
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_end = i + batch_size
        collection.upsert(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )

    print(f"Stored {len(texts)} chunks in collection '{COLLECTION_NAME}'.")


def build_vector_store(reset: bool = True) -> chromadb.Collection:
    """End-to-end: chunk documents, embed, and store in ChromaDB."""
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} documents.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    collection = get_collection(reset=reset)
    embed_and_store(chunks, collection, model)
    return collection


def load_retriever():
    """
    Load the embedding model and ChromaDB collection for query time.
    Returns (model, collection).
    """
    model = SentenceTransformer(EMBED_MODEL_NAME)
    collection = get_collection(reset=False)
    return model, collection


if __name__ == "__main__":
    collection = build_vector_store(reset=True)
    print(f"\nCollection count: {collection.count()} chunks indexed.")
    print("\n--- Test retrieval: 'how hard is Hilfinger CS61B' ---")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    query_embedding = model.encode(["how hard is Hilfinger CS61B"], convert_to_list=True)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"\nSource: {meta['source']} | Distance: {dist:.4f}")
        print(f"Text: {doc[:250]}...")
