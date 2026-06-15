"""
ingest.py — Document loading and chunking pipeline.

Loads .txt files from the /documents/ directory, cleans them,
and splits them into chunks using a paragraph-aware strategy
(600-character chunks with 100-character overlap).
"""

import os
import re
from pathlib import Path


DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_documents(docs_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """
    Load all .txt files from docs_dir.
    Returns a list of dicts: {source, text}.
    """
    documents = []
    for filepath in sorted(docs_dir.glob("*.txt")):
        with open(filepath, encoding="utf-8") as f:
            raw = f.read()
        text = clean_text(raw)
        if text:
            documents.append({"source": filepath.name, "text": text})
    return documents


def clean_text(text: str) -> str:
    """
    Remove common artifacts: excess blank lines, trailing whitespace,
    HTML entities, and non-printable characters.
    """
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'").replace("&quot;", '"')

    # Remove carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse more than 2 consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace from every line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def split_into_chunks(text: str, source: str,
                      chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Paragraph-aware chunking strategy:
    1. Split on double newlines (paragraph boundaries).
    2. If a paragraph fits within chunk_size, keep it whole.
    3. If a paragraph is too long, split with fixed-size/overlap.
    Returns a list of dicts: {text, source, chunk_index}.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph alone is larger than chunk_size, split it directly
        if len(para) > chunk_size:
            # First flush whatever is in current_chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                current_chunk = ""
            # Fixed-size split of the large paragraph
            for fixed_chunk in _fixed_size_split(para, chunk_size, overlap):
                chunks.append({
                    "text": fixed_chunk,
                    "source": source,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
            continue

        # Would adding this paragraph overflow the current chunk?
        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Save current chunk and start a new one with overlap
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                # Overlap: carry the tail of the previous chunk
                overlap_text = current_chunk[-overlap:].strip() if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

    # Flush any remaining text
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "source": source,
            "chunk_index": chunk_index,
        })

    return chunks


def _fixed_size_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long string into fixed-size windows with overlap."""
    result = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        result.append(text[start:end])
        start += chunk_size - overlap
    return result


def build_chunks(docs_dir: Path = DOCUMENTS_DIR) -> list[dict]:
    """Full pipeline: load → clean → chunk. Returns all chunks."""
    documents = load_documents(docs_dir)
    all_chunks = []
    for doc in documents:
        chunks = split_into_chunks(doc["text"], doc["source"])
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"\nTotal chunks produced: {len(chunks)}")
    print(f"Documents processed: {len(set(c['source'] for c in chunks))}")
    print("\n--- 5 Sample Chunks ---\n")
    import random
    sample = random.sample(chunks, min(5, len(chunks)))
    for i, chunk in enumerate(sample, 1):
        print(f"[{i}] Source: {chunk['source']} | Index: {chunk['chunk_index']}")
        print(f"    Length: {len(chunk['text'])} chars")
        print(f"    Text: {chunk['text'][:200]}...")
        print()
