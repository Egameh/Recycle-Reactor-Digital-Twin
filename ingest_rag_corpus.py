"""
RAG Corpus Ingestion for the TEP Digital Twin Agent

Extracts text from PDF/Markdown source documents, chunks it, embeds each
chunk with Gemini's embedding model, and saves the result for retrieval.

Requires:
    pip install google-genai pypdf
    export GOOGLE_API_KEY=your_key_here

Usage:
    python ingest_rag_corpus.py /path/to/data/folder file1.pdf file2.pdf file3.md

Example:
    python ingest_rag_corpus.py "/Users/egmh/Downloads/Tennesy Folder/TEP_data" \\
        "/Users/egmh/Downloads/TEP_pdf.pdf" \\
        "/Users/egmh/Downloads/TEP_pdf2.pdf" \\
        "/Users/egmh/Downloads/tep_plant_overview.md"

Output (saved into the data folder):
    rag_embeddings.npy   -- one embedding vector per chunk
    rag_metadata.json    -- chunk text + source + chunk index, same order as embeddings
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CHUNK_WORD_SIZE = 250
CHUNK_OVERLAP_WORDS = 50
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768


def extract_text_from_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    elif path.suffix.lower() in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path.name})")


def chunk_text(text: str, source_name: str) -> list:
    """
    Splits text into overlapping word-count chunks. Simple and robust across
    document types; good enough for retrieval at this corpus size.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + CHUNK_WORD_SIZE
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        if chunk_str.strip():
            chunks.append({
                "source": source_name,
                "chunk_index": chunk_index,
                "text": chunk_str,
            })
            chunk_index += 1

        start += CHUNK_WORD_SIZE - CHUNK_OVERLAP_WORDS

    return chunks


def embed_chunks(client, chunks: list) -> np.ndarray:
    from google.genai import types

    embeddings = np.zeros((len(chunks), EMBEDDING_DIM), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk["text"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        embeddings[i] = result.embeddings[0].values
        print(f"  Embedded chunk {i + 1}/{len(chunks)} ({chunk['source']}, "
              f"chunk {chunk['chunk_index']})")
        time.sleep(0.1)  # light rate-limit courtesy

    return embeddings


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_rag_corpus.py /path/to/data/folder file1.pdf file2.pdf ...")
        sys.exit(1)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: set your GOOGLE_API_KEY environment variable first.")
        sys.exit(1)

    from google import genai

    data_dir = Path(sys.argv[1])
    source_paths = [Path(p) for p in sys.argv[2:]]

    all_chunks = []
    for path in source_paths:
        if not path.exists():
            print(f"Warning: file not found, skipping: {path}")
            continue
        print(f"Extracting text from {path.name}...")
        text = extract_text(path)
        chunks = chunk_text(text, source_name=path.name)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    print("Embedding chunks (this may take a few minutes)...\n")

    client = genai.Client()
    embeddings = embed_chunks(client, all_chunks)

    np.save(data_dir / "rag_embeddings.npy", embeddings)
    with open(data_dir / "rag_metadata.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nSaved {len(all_chunks)} embedded chunks to:")
    print(f"  {data_dir / 'rag_embeddings.npy'}")
    print(f"  {data_dir / 'rag_metadata.json'}")
