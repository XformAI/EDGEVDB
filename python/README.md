# EdgeVDB Python SDK

> **Python wrapper for EdgeVDB on-device vector database with ctypes FFI binding.**

The EdgeVDB Python SDK provides a Pythonic interface to the EdgeVDB C++ core library using ctypes. It enables Python applications to use EdgeVDB's vector database capabilities on desktop and Raspberry Pi platforms.

## Features

- **ctypes FFI Binding** — Direct calls to C API with no Python dependencies
- **Context Manager Support** — Automatic resource cleanup with `with` statements
- **Type Hints** — Full type annotations for IDE support
- **Zero Python Dependencies** — Only standard library and ctypes
- **Cross-Platform** — Linux, macOS, Windows, Raspberry Pi
- **Flexible Embedding** — Use any embedding provider, or the built-in embedder with real ONNX Runtime inference (dynamically loaded; check `Embedder.is_semantic`)
- **Retrieval Modes (0.2.0)** — `retrieval_mode=0` hybrid (vector ∪ BM25, default), `1` vector-only, `2` BM25-only; `db.query_bm25(text)` needs no embedding model at all
- **Tunable RAG stack (0.2.0)** — `enable_page_index`, `ranker_mode` (linear/RRF/graph-RRF), opt-in HNSW modes (`hnsw_use_heuristic_selection`, `hnsw_adaptive_ef`, `hnsw_quantized_search`) with automatic recall-guarded fallback
- **Multi-Device Sync** — `edgevdb.sync.SyncHelper` file-based delta exchange

### 30-second example

```python
from edgevdb import EdgeVDB

# Lexical search, zero models required:
with EdgeVDB("./data", retrieval_mode=2) as db:
    db.insert_chunk("the zephyrium alloy datasheet", [0.0] * 384, doc_id=1)
    print(db.query_bm25("zephyrium", top_k=3)[0].text)
```

## Installation

### From PyPI (Recommended)

```bash
pip install edgevdb
```

Pre-built wheels include native libraries for **Linux** (x86_64, glibc 2.28+), **macOS** (arm64/x86_64), and **Windows** (x86_64).

### From Source

```bash
# Build the C++ core first
cd ..
cmake --preset desktop-release
cmake --build build/desktop-release

# Copy shared library to Python package (platform-specific)
# Linux:
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/lib/linux/
# macOS:
# cp build/desktop-release/core/libedgevdb_shared.dylib python/edgevdb/lib/darwin/
# Windows:
# copy build\desktop-release\core\edgevdb_shared.dll python\edgevdb\lib\windows\

# Install in development mode
cd python
pip install -e .
```

## Quick Start

### Without ONNX (Recommended)

Use embeddings from any provider (OpenAI, Cohere, sentence-transformers, etc.):

```python
from edgevdb import EdgeVDB

# Open database
db = EdgeVDB("./my_database")

# Get embeddings from your preferred provider
# Example with sentence-transformers:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Machine learning finds patterns in data")

# Insert with pre-computed embedding
chunk_id = db.insert_chunk(
    text="Machine learning finds patterns in data",
    embedding=embedding,
    doc_id=1,
    page_number=0
)

# Query
query_emb = model.encode("what is ML?")
results = db.query_vector(query_emb, query_text="what is ML?", top_k=5)

for r in results:
    print(f"score={r.score:.3f} text={r.text}")

# Object store
doc_id = db.put_object("Document", {"title": "ML Intro", "author": "Alice"})
db.add_relation("has_chunk", doc_id, chunk_id)

db.save()
db.close()
```

### With Built-in Embedder

```python
from edgevdb import EdgeVDB, Embedder

# Create embedder
embedder = Embedder(
    model_path="models/model.onnx",
    vocab_path="models/vocab.txt",
    threads=2
)

# Use with context manager
with EdgeVDB("./my_database") as db:
    # Auto-embed on insert
    chunk_id = db.insert_text(
        embedder, 
        "Deep learning uses neural networks",
        doc_id=1,
        page_number=0
    )

    # Auto-embed on query
    results = db.query_text(embedder, "neural network architecture", top_k=5)
    print(results.context_string)
```

## API Reference

### EdgeVDB

Main database class.

#### Constructor

```python
EdgeVDB(storage_dir: str, **kwargs)
```

**Parameters:**
- `storage_dir` (str): Directory for database files
- `hnsw_M` (int): HNSW M parameter (default: 16)
- `hnsw_ef_construction` (int): HNSW ef_construction (default: 200)
- `hnsw_ef_search` (int): HNSW ef_search (default: 64)
- `ranker_alpha` (float): Cosine weight (default: 0.70)
- `ranker_beta` (float): Page proximity weight (default: 0.20)
- `ranker_gamma` (float): Keyword weight (default: 0.10)
- `token_budget` (int): Max tokens in context (default: 3200)
- `embedding_threads` (int): ONNX thread count (default: 2)
- `enable_knowledge_graph` (bool): Enable KG (default: True)
- `enable_sync` (bool): Enable sync (default: False)
- `device_id` (str): Device ID for sync (default: auto-generated)

#### Methods

##### Vector Store

**insert_chunk(text, embedding, doc_id=0, page_number=0) -> int**
- Insert text with pre-computed embedding
- Returns chunk ID

```python
chunk_id = db.insert_chunk(
    text="Your text here",
    embedding=[0.1, 0.2, ...],  # 384-dim float array
    doc_id=1,
    page_number=0
)
```

**insert_text(embedder, text, doc_id=0, page_number=0) -> int**
- Insert text with auto-embedding via embedder
- Returns chunk ID

```python
chunk_id = db.insert_text(
    embedder,
    "Your text here",
    doc_id=1,
    page_number=0
)
```

**remove_chunk(chunk_id)**
- Remove chunk by ID

```python
db.remove_chunk(chunk_id)
```

**query_vector(embedding, query_text="", top_k=5) -> QueryResults**
- Query with pre-computed embedding
- Returns QueryResults object

```python
results = db.query_vector(
    embedding=[0.1, 0.2, ...],
    query_text="search query",
    top_k=5
)
```

**query_text(embedder, query, top_k=5, use_kg_expansion=False) -> QueryResults**
- Query with auto-embedding via embedder
- Returns QueryResults object

```python
results = db.query_text(
    embedder,
    "search query",
    top_k=5,
    use_kg_expansion=False
)
```

##### Object Store

**put_object(type_name, properties) -> int**
- Store JSON object
- Returns object ID

```python
doc_id = db.put_object(
    "Document",
    {"title": "My Doc", "author": "Alice"}
)
```

**get_object(object_id) -> Optional[Dict]**
- Retrieve object by ID
- Returns dict or None if not found

```python
obj = db.get_object(doc_id)
if obj:
    print(obj["title"])
```

**remove_object(object_id)**
- Soft delete object

```python
db.remove_object(doc_id)
```

##### Relations

**add_relation(name, from_id, to_id)**
- Add typed edge between objects

```python
db.add_relation("has_chunk", doc_id, chunk_id)
```

##### Lifecycle

**save()**
- Flush all data to disk

```python
db.save()
```

**close()**
- Release native resources

```python
db.close()
```

**Context Manager**

```python
with EdgeVDB("./data") as db:
    # Auto-save and close on exit
    db.insert_chunk("text", embedding, doc_id=1)
```

### Embedder

ONNX embedding model wrapper.

#### Constructor

```python
Embedder(model_path: str, vocab_path: str, threads: int = 2)
```

**Parameters:**
- `model_path` (str): Path to ONNX model file
- `vocab_path` (str): Path to vocabulary file
- `threads` (int): Number of inference threads (default: 2)

#### Methods

**embed(text: str) -> List[float]**
- Embed text to 384-dim vector
- Returns list of floats

```python
embedding = embedder.embed("Hello world")
```

**destroy()**
- Release native resources

```python
embedder.destroy()
```

### QueryResults

Query result container with lazy access.

#### Properties

**count** (int): Number of results

```python
print(f"Found {results.count} results")
```

**context_string** (str): Pre-assembled RAG context

```python
print(results.context_string)
```

#### Methods

**__getitem__(index) -> ChunkResult**
- Access individual result by index

```python
result = results[0]
print(result.text)
```

**__iter__()**
- Iterate over results

```python
for r in results:
    print(f"{r.score}: {r.text}")
```

**to_list() -> List[ChunkResult]**
- Convert to list

```python
results_list = results.to_list()
```

**free()**
- Free native query handle (called automatically by __del__)

```python
results.free()
```

### ChunkResult

Single query result.

#### Attributes

- **chunk_id** (int): Unique chunk identifier
- **text** (str): Chunk text content
- **score** (float): Hybrid similarity score [0.0, 1.0]
- **page_number** (int): Page number in document
- **doc_id** (int): Document identifier

```python
for r in results:
    print(f"ID: {r.chunk_id}")
    print(f"Text: {r.text}")
    print(f"Score: {r.score:.3f}")
    print(f"Page: {r.page_number}")
```

## Examples

### RAG Pipeline

```python
from edgevdb import EdgeVDB
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
db = EdgeVDB("./rag_database")

# Index documents
documents = [
    {"id": 1, "text": "Python is a high-level programming language."},
    {"id": 2, "text": "Machine learning is a subset of AI."},
    {"id": 3, "text": "Vector databases enable semantic search."},
]

for doc in documents:
    embedding = model.encode(doc["text"])
    db.insert_chunk(doc["text"], embedding, doc_id=doc["id"])

# Query
query = "What is semantic search?"
query_emb = model.encode(query)
results = db.query_vector(query_emb, query_text=query, top_k=2)

# Assemble context
context = results.context_string
print(f"Context: {context}")

db.save()
db.close()
```

### Object Store + Relations

```python
from edgevdb import EdgeVDB

db = EdgeVDB("./my_database")

# Store documents
doc1_id = db.put_object("Document", {
    "title": "Introduction to ML",
    "author": "Alice",
    "year": 2024
})

doc2_id = db.put_object("Document", {
    "title": "Advanced Topics",
    "author": "Bob",
    "year": 2024
})

# Store chunks with embeddings
chunk1_id = db.insert_chunk("ML is fascinating", emb, doc_id=doc1_id)
chunk2_id = db.insert_chunk("Deep learning is powerful", emb, doc_id=doc2_id)

# Link chunks to documents
db.add_relation("has_chunk", doc1_id, chunk1_id)
db.add_relation("has_chunk", doc2_id, chunk2_id)

db.save()
db.close()
```

### Error Handling

```python
from edgevdb import EdgeVDB, set_log_level

# Enable debug logging
set_log_level(3)

try:
    db = EdgeVDB("./my_database")
    
    # Operations
    chunk_id = db.insert_chunk("text", embedding, doc_id=1)
    
    # Object not found returns None (doesn't throw)
    obj = db.get_object(999)
    if obj is None:
        print("Object not found")
    
    db.save()
    db.close()
    
except RuntimeError as e:
    print(f"EdgeVDB error: {e}")
```

## Library Discovery

The Python SDK automatically searches for the EdgeVDB shared library in the following locations:

1. Platform-specific directory (`edgevdb/lib/<platform>/`) — **preferred**
2. Package lib directory (`edgevdb/lib/`)
3. Package directory (`edgevdb/`)
4. Current working directory
5. `build/desktop-release/core/`
6. `build/desktop-debug/core/`

**Library Layout:**
```
python/edgevdb/lib/
  linux/    → libedgevdb_shared.so
  darwin/   → libedgevdb_shared.dylib
  windows/  → edgevdb_shared.dll, libedgevdb_shared.dll
```

## Performance Considerations

### Embedding Provider Choice

| Provider | Speed | Quality | Offline | Cost |
|----------|-------|--------|---------|------|
| sentence-transformers | Fast | Good | ✅ | Free |
| OpenAI API | Slow | Excellent | ❌ | Paid |
| Cohere API | Medium | Good | ❌ | Paid |
| Built-in ONNX | Medium | Good | ✅ | Free |

### Batch Operations

For large-scale operations, consider batching:

```python
# Batch insert
embeddings = model.encode(texts)
for text, emb in zip(texts, embeddings):
    db.insert_chunk(text, emb, doc_id=doc_id)

db.save()  # Save once after all inserts
```

### Memory Management

- Query results hold native handles; call `results.free()` or use context manager
- Embedders hold native resources; call `embedder.destroy()` when done
- Database handles are released by `close()` or context manager

## Platform-Specific Notes

### Linux

```bash
# Build
cmake --preset desktop-release
cmake --build build/desktop-release

# Install
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/lib/linux/
pip install -e python/
```

### macOS

```bash
# Build
cmake --preset desktop-release
cmake --build build/desktop-release

# Install
cp build/desktop-release/core/libedgevdb_shared.dylib python/edgevdb/lib/darwin/
pip install -e python/
```

### Windows

```powershell
# Build
cmake --preset desktop-release
cmake --build build/desktop-release

# Install
copy build\desktop-release\core\edgevdb_shared.dll python\edgevdb\lib\windows\
pip install -e python\
```

### Raspberry Pi

```bash
# Build with NEON support
cmake --preset desktop-release
cmake --build build/desktop-release

# Install
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/lib/linux/
pip install -e python/
```

## Testing

```bash
cd python

# Run tests
python -m unittest tests.test_edgevdb -v

# Or with pytest
pytest tests/ -v
```

## Troubleshooting

### Library Not Found

**Error:** `FileNotFoundError: Could not find EdgeVDB library`

**Solution:**
1. Build the C++ core: `cmake --preset desktop-release && cmake --build build/desktop-release`
2. Copy the shared library to `python/edgevdb/lib/<platform>/`
3. Verify the library name matches your platform

### Import Errors

**Error:** `ImportError: dynamic module does not define init function`

**Solution:**
- Ensure the shared library was built for your platform
- Check Python architecture matches library (32-bit vs 64-bit)
- Rebuild the C++ core for your platform

### Segmentation Faults

**Error:** Python crashes with segmentation fault

**Solution:**
- Ensure you're using the correct library version
- Check that you're not accessing freed handles
- Verify embedding dimensions are exactly 384
- Enable debug logging: `set_log_level(3)`

## Contributing

### Development Setup

```bash
# Build C++ core in debug mode
cmake --preset desktop-debug
cmake --build build/desktop-debug

# Copy debug library
cp build/desktop-debug/core/libedgevdb_shared.so python/edgevdb/

# Install in development mode
cd python
pip install -e .
```

### Running Tests

```bash
cd python
python -m unittest tests.test_edgevdb -v
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public APIs
- Run black and flake8

## See Also

- [../README.md](../README.md) — Project overview
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
- [../../docs/python_integration.md](../../docs/python_integration.md) — Python integration guide
- [examples/](examples/) — Example scripts
