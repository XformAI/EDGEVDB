# Python Integration Guide

## Prerequisites
- Python 3.8+
- Built EdgeVDB shared library (`libedgevdb_shared.so` / `.dylib` / `.dll`)

## Building the Library
```bash
cmake --preset desktop-release
cmake --build --preset desktop-release
```

The shared library will be in `build/desktop-release/core/`.

## Setup
```bash
# Copy library to Python package
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/

# Or set the library path
export LD_LIBRARY_PATH=build/desktop-release/core:$LD_LIBRARY_PATH
```

## Usage
```python
from edgevdb import EdgeVDB, Embedder

# Open database
with EdgeVDB("./data") as db:
    # Create embedder
    embedder = Embedder("model.onnx", "vocab.txt")

    # Insert
    chunk_id = db.insert_text(embedder, "Hello world", doc_id=1, page_number=0)
    print(f"Inserted chunk {chunk_id}")

    # Query
    results = db.query_text(embedder, "hello", top_k=5)
    for r in results:
        print(f"  [{r.score:.3f}] {r.text}")

    # Context string for LLM
    context = results.context_string
    print(f"\nRAG context:\n{context}")

    # Object store
    obj_id = db.put_object("Document", {"title": "Test", "author": "Alice"})
    obj = db.get_object(obj_id)
    print(f"Object: {obj}")

    # Relations
    db.add_relation("authored_by", obj_id, 100)
```

## API Reference

### `EdgeVDB(storage_dir, **config)`
Open or create a database.

### `Embedder(model_path, vocab_path, threads=2)`
Create an embedding model.

### `db.insert_text(embedder, text, doc_id, page_number)` → chunk_id
### `db.query_text(embedder, query, top_k=5)` → QueryResults
### `db.put_object(type_name, properties)` → object_id
### `db.get_object(id)` → dict or None
### `db.add_relation(name, from_id, to_id)`
