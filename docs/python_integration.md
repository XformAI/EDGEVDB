# Python Integration Guide

> **Comprehensive guide for integrating EdgeVDB into Python applications.**

## Prerequisites

- **Python** 3.8+
- Built EdgeVDB shared library: `.so` (Linux), `.dylib` (macOS), `.dll` (Windows)
- ctypes (standard library, no pip install needed)

## Building the Library

```bash
cmake --preset desktop-release
cmake --build build/desktop-release
```

**Output:** `build/desktop-release/core/libedgevdb_shared.so` (Linux), `.dylib` (macOS), `.dll` (Windows)

## Setup

### Option A: Development Mode (Recommended)

```bash
# Copy shared library into the Python package
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/lib/linux/

# Install in development mode
cd python
pip install -e .
```

### Option B: Library Search Path

```bash
# Set library search path
export LD_LIBRARY_PATH=build/desktop-release/core:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=build/desktop-release/core:$DYLD_LIBRARY_PATH  # macOS
# On Windows: Add to PATH
```

### Option C: System-wide Installation

```bash
# Copy library to system location
sudo cp build/desktop-release/core/libedgevdb_shared.so /usr/local/lib/  # system-wide
# Or into package: cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/lib/linux/
sudo ldconfig  # Linux only

# Install Python package
cd python
pip install -e .
```

## Usage

### Without ONNX (Recommended)

Use embeddings from **any provider** (OpenAI, Cohere, sentence-transformers, etc.):

> **Note:** `insert_chunk(text, embedding, ...)` takes **text first**, then embedding — matching the C API order.

```python
from edgevdb import EdgeVDB

with EdgeVDB("./data") as db:
    # Get embedding from your preferred provider
    embedding = your_provider.embed("Machine learning finds patterns")

    # Insert: text first, then embedding
    chunk_id = db.insert_chunk(
        "Machine learning finds patterns",
        embedding,
        doc_id=1,
        page_number=0
    )

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

### With Built-in Embedder

```python
from edgevdb import EdgeVDB, Embedder

with EdgeVDB("./data") as db:
    embedder = Embedder("model.onnx", "vocab.txt", threads=2)

    chunk_id = db.insert_text(embedder, "Hello world", doc_id=1, page_number=0)

    results = db.query_text(embedder, "hello", top_k=5)
    for r in results:
        print(f"  [{r.score:.3f}] {r.text}")
    print(results.context_string)
    results.free()  # Free native query handle

    embedder.destroy()  # Free native embedder handle
```

### Object Store & Relations

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

### Async Pattern

```python
import asyncio
from edgevdb import EdgeVDB

async def async_insert(db, text, embedding):
    # Run blocking operations in thread pool
    loop = asyncio.get_event_loop()
    chunk_id = await loop.run_in_executor(
        None,
        db.insert_chunk,
        text,
        embedding,
        1,
        0
    )
    return chunk_id

async def main():
    with EdgeVDB("./data") as db:
        embedding = [0.1] * 384
        chunk_id = await async_insert(db, "Test text", embedding)
        print(f"Inserted: {chunk_id}")

asyncio.run(main())
```

## Configuration

### Database Configuration

```python
from edgevdb import EdgeVDB

db = EdgeVDB(
    "./data",
    hnsw_M=16,
    hnsw_ef_construction=200,
    hnsw_ef_search=64,
    ranker_alpha=0.70,
    ranker_beta=0.20,
    ranker_gamma=0.10,
    token_budget=3200,
    enable_knowledge_graph=True,
    enable_sync=False
)
```

### HNSW Parameters

```python
db = EdgeVDB(
    "./data",
    hnsw_M=16,              # Max connections per node
    hnsw_ef_construction=200,  # Build-time search width
    hnsw_ef_search=64       # Query-time search width
)
```

### Ranking Weights

```python
db = EdgeVDB(
    "./data",
    ranker_alpha=0.70,    # Cosine similarity
    ranker_beta=0.20,     # Page proximity
    ranker_gamma=0.10     # Keyword overlap
)
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
| `EdgeVDB(storage_dir, **kwargs)` | Open or create database |
| `db.insert_chunk(text, embedding, doc_id, page_number)` | Insert with pre-computed embedding |
| `db.insert_text(embedder, text, doc_id, page_number)` | Insert with auto-embedding |
| `db.remove_chunk(chunk_id)` | Remove chunk by ID |
| `db.query_vector(embedding, query_text, top_k)` | Query with pre-computed embedding |
| `db.query_text(embedder, query, top_k, use_kg_expansion)` | Query with auto-embedding |
| `results.context_string` | Pre-assembled RAG context string |
| `results.free()` | Free native query handle — always call after use |
| `db.put_object(type_name, properties)` | Store a JSON object |
| `db.get_object(id)` | Retrieve object by ID |
| `db.remove_object(id)` | Soft-delete object |
| `db.add_relation(name, from_id, to_id)` | Add a relation |
| `db.save()` / `db.close()` | Persist / cleanup |
| `Embedder(model_path, vocab_path, threads)` | Create embedder |
| `embedder.destroy()` | Free embedder handle |
| `version()` / `set_log_level(level)` | Utilities |

## Performance Optimization

### Batch Operations

```python
# Batch insert for efficiency
texts = ["text1", "text2", "text3"]
embeddings = [provider.embed(t) for t in texts]

for text, emb in zip(texts, embeddings):
    db.insert_chunk(text, emb, doc_id=1, page_number=0)

db.save()  # Save once after all inserts
```

### Use Context Managers

```python
# Automatic cleanup
with EdgeVDB("./data") as db:
    # Operations...
    pass  # Auto-save and close on exit
```

### Thread Pool for Blocking Operations

```python
from concurrent.futures import ThreadPoolExecutor

def process_document(db, text, doc_id):
    with ThreadPoolExecutor() as executor:
        # Run blocking operations in pool
        future = executor.submit(db.insert_chunk, text, embedding, doc_id, 0)
        return future.result()
```

## Troubleshooting

### Library Not Found

**Error:** `FileNotFoundError: Could not find EdgeVDB library`

**Solutions:**
- Copy the shared library to `python/edgevdb/`
- Set `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS)
- Verify the library name matches your platform
- Check that the library was built for your architecture

### Import Errors

**Error:** `ImportError: dynamic module does not define init function`

**Solutions:**
- Ensure the shared library was built for your platform
- Check Python architecture matches library (32-bit vs 64-bit)
- Rebuild the C++ core for your platform

### Segmentation Faults

**Error:** Python crashes with segmentation fault

**Solutions:**
- Ensure you're using the correct library version
- Check that you're not accessing freed handles
- Verify embedding dimensions are exactly 384
- Enable debug logging: `set_log_level(3)`

## Testing

### Unit Tests

```bash
cd python

# Run tests with unittest
python3 -m unittest tests.test_edgevdb -v

# Or with pytest
pytest tests/ -v
```

### Manual Testing

```python
from edgevdb import EdgeVDB

# Test basic operations
db = EdgeVDB("./test_db")
embedding = [0.1] * 384

# Insert
chunk_id = db.insert_chunk("Test text", embedding, doc_id=1, page_number=0)
print(f"Inserted: {chunk_id}")

# Query
results = db.query_vector(embedding, query_text="test", top_k=5)
print(f"Found {results.count} results")

# Cleanup
results.free()
db.save()
db.close()
```

## Best Practices

1. **Context Managers**: Use `with EdgeVDB()` for automatic cleanup
2. **Resource Cleanup**: Always call `results.free()` after processing results
3. **Batch Operations**: Minimize database save operations by batching inserts
4. **Error Handling**: Check return values and handle exceptions
5. **Library Path**: Ensure the shared library is in the correct location
6. **Thread Safety**: Use thread pools for concurrent operations
7. **Logging**: Enable debug logging during development

## See Also

- [../../python/README.md](../../python/README.md) — Python SDK documentation
- [../architecture.md](architecture.md) — Architecture overview
- [../api_reference.md](api_reference.md) — C API reference
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
