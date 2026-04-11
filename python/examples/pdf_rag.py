#!/usr/bin/env python3
"""
Example: PDF RAG pipeline with EdgeVDB.

Usage:
    python pdf_rag.py document.pdf "What is the main topic?"
"""

import sys
import os

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extract text from PDF pages.
    Requires: pip install PyMuPDF
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                pages.append((page_num, text.strip()))
        doc.close()
        return pages
    except ImportError:
        print("Install PyMuPDF: pip install PyMuPDF")
        sys.exit(1)


def chunk_text(text: str, max_chars: int = 500) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            # 20% overlap
            overlap = max(1, len(current) // 5)
            current = current[-overlap:]
            current_len = sum(len(w) + 1 for w in current)
        current.append(word)
        current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <pdf_path> <query>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    query = sys.argv[2]

    from edgevdb import EdgeVDB, Embedder

    # Setup
    data_dir = "./edgevdb_data"
    model_path = os.path.join(os.path.dirname(__file__), "../../models/model.onnx")
    vocab_path = os.path.join(os.path.dirname(__file__), "../../models/vocab.txt")

    print(f"Opening EdgeVDB at {data_dir}...")
    db = EdgeVDB(data_dir)
    embedder = Embedder(model_path, vocab_path)

    # Extract and ingest
    print(f"Processing {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    total_chunks = 0

    for page_num, page_text in pages:
        chunks = chunk_text(page_text)
        for chunk in chunks:
            db.insert_text(embedder, chunk, doc_id=1, page_number=page_num)
            total_chunks += 1

    print(f"Ingested {total_chunks} chunks from {len(pages)} pages")
    db.save()

    # Query
    print(f"\nQuery: {query}")
    results = db.query_text(embedder, query, top_k=5)

    print(f"\n--- Top {results.count} Results ---")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r.score:.3f} | Page: {r.page_number}")
        print(f"    {r.text[:200]}...")

    print(f"\n--- RAG Context ---")
    print(results.context_string[:2000])

    db.close()


if __name__ == "__main__":
    main()
