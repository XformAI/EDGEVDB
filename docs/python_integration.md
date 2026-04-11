# Python Integration Guide

## Prerequisites
- Python 3.8+
- Built EdgeVDB shared library: `.so` (Linux), `.dylib` (macOS), `.dll` (Windows)

## Building the Library

```bash
cmake --preset desktop-release
cmake --build build/desktop-release
```

Output: `build/desktop-release/core/libedgevdb_shared.so`

## Setup

```bash
# Copy shared library into the Python package
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/

# Install in development mode
cd python && pip install -e .
```

Or set the library search path:
```bash
export LD_LIBRARY_PATH=build/desktop-release/core:$LD_LIBRARY_PATH
```

## Usage — Without ONNX (Recommended)

Use embeddings from **any provider** (OpenAI, Cohere, sentence-transformers, etc.):

> **Note:** `insert_chunk(text, embedding, ...)` takes **text first**, then embedding — matching the C API order.

```python
from edgevdb import EdgeVDB

with EdgeVDB("./data") as db:
    # Get embedding from your preferred provider
    embedding = your_provider.embed("Machine learning finds patterns")

    # Insert: text first, then embedding
    chunk_id = db.insert_chunk("Machine learning finds patterns", embedding,
                               doc_id=1, page_number=0)

    # Query with pre-computed embedding
    query_emb = your_provider.embed("what is ML?")
    results = db.query_vector(query_emb, query_text="what is ML?", top_k=5)

    # Each result has .chunk_id, .text, .score, .page_number
    for r in results:
        print(f"  [{r.score:.3f}] {r.text}")

    # RAG context string (pre-assembled for LLM)
    print(results.context_string)

    # Free the native query handle when done
    results.free()
```

## Usage — With Built-in Embedder

```python
from edgevdb import EdgeVDB, Embedder

with EdgeVDB("./data") as db:
    embedder = Embedder("model.onnx", "vocab.txt")

    chunk_id = db.insert_text(embedder, "Hello world", doc_id=1, page_number=0)

    results = db.query_text(embedder, "hello", top_k=5)
    for r in results:
        print(f"  [{r.score:.3f}] {r.text}")
    print(results.context_string)
    results.free()  # Free native query handle

    embedder.destroy()  # Free native embedder handle
```

## Object Store & Relations

```python
from edgevdb import EdgeVDB

with EdgeVDB("./data") as db:
    # Objects
    obj_id = db.put_object("Document", {"title": "ML Guide", "author": "Alice"})
    obj = db.get_object(obj_id)  # Returns dict or None
    print(obj)

    # Relations
    db.add_relation("authored_by", obj_id, 100)
```

## Utilities

```python
from edgevdb import version, set_log_level

print(f"EdgeVDB v{version()}")  # "1.0.0"
set_log_level(2)  # 0=off, 1=error, 2=info, 3=debug
```

## API Summary

| Method | Description |
|--------|-------------|
| `EdgeVDB(storage_dir)` | Open or create database |
| `db.insert_chunk(text, embedding, doc_id, page_number)` | Insert with pre-computed embedding |
| `db.insert_text(embedder, text, doc_id, page_number)` | Insert with auto-embedding |
| `db.query_vector(embedding, query_text, top_k)` | Query with pre-computed embedding |
| `db.query_text(embedder, query, top_k)` | Query with auto-embedding |
| `results.context_string` | Pre-assembled RAG context string |
| `results.free()` | Free native query handle — always call after use |
| `db.put_object(type_name, properties)` | Store a JSON object |
| `db.get_object(id)` | Retrieve object by ID |
| `db.add_relation(name, from_id, to_id)` | Add a relation |
| `db.save()` / `db.close()` | Persist / cleanup |
| `Embedder(model_path, vocab_path)` | Create embedder |
| `version()` / `set_log_level(level)` | Utilities |

## Running Tests

```bash
cp build/desktop-debug/core/libedgevdb_shared.so python/edgevdb/
cd python && python3 -m unittest tests.test_edgevdb -v
# Or: cd python && pytest tests/ -v
```
