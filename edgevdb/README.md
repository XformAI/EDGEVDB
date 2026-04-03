# EdgeVDB SDK

> **Embeddable cross-platform vector database with HNSW ANN, hybrid retrieval, knowledge graph, relational object store, and CRDT-based sync — all in a single zero-dependency C++ library.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](CHANGELOG.md)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()
[![Platforms](https://img.shields.io/badge/platforms-Android%20%7C%20iOS%20%7C%20Desktop-lightgrey.svg)]()

## Features

| Feature | Description |
|---|---|
| **HNSW ANN Index** | 384-dim float32 vectors, M=16, ef=200/64 |
| **Hybrid Retrieval** | α·cosine + β·page_proximity + γ·keyword |
| **Knowledge Graph** | On-device NER → entity graph → multi-hop expansion |
| **Embedding Pipeline** | WordPiece tokenizer → ONNX MiniLM → 384-dim L2-norm |
| **Object Store** | Schema-less NoSQL with typed property indexing |
| **Relation Index** | Foreign key edges between objects |
| **CRDT Sync** | LWW vector clock sync across devices |
| **Stable C API** | `vectordb.h` — single header, ABI-stable |
| **Platform SDKs** | Android (JNI/Kotlin), iOS (Swift), Python (ctypes) |

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       EdgeVDB SDK                          │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Vector DB   │  │  Relational  │  │   Data Sync     │  │
│  │  (HNSW ANN)  │  │  Object DB   │  │   Engine        │  │
│  │  Hybrid Rank │  │  (NoSQL ORM) │  │   (CRDTs)       │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       Embedding Pipeline (Text → Vector)             │  │
│  │  Tokenizer → ONNX MiniLM → L2-norm 384-dim vectors  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Pure C Public API (vectordb.h)          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │ Android    │  │ iOS Swift  │  │ Python (Desktop/Pi)  │  │
│  │ JNI/Kotlin │  │ Wrapper    │  │ ctypes harness       │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## Quick Start

### Android (Kotlin)
```kotlin
val db = EdgeVDB.open(context)
val embedder = Embedder.fromAssets(context)

// Insert
db.insertText(embedder, "Neural networks are computational models...", docId = 1, pageNumber = 0)

// Query
val results = db.queryText(embedder, "What are neural networks?", topK = 5)
println(results.contextString)

db.close()
```

### iOS (Swift)
```swift
let db = try EdgeVDB(storageDir: docsURL)
let embedder = try Embedder(modelResource: "model", vocabResource: "vocab")

let chunkId = try db.insertText(using: embedder, text: "...", docId: 1, pageNumber: 0)
let results = try db.queryText(using: embedder, query: "search query", topK: 5)
print(results.contextString)
```

### Python
```python
from edgevdb import EdgeVDB, Embedder

with EdgeVDB("./data") as db:
    embedder = Embedder("model.onnx", "vocab.txt")
    db.insert_text(embedder, "Hello world", doc_id=1, page_number=0)
    results = db.query_text(embedder, "hello", top_k=5)
    print(results.context_string)
```

### C API
```c
EvdbConfig config;
evdb_default_config(&config);
config.storage_dir = "./data";

EvdbHandle* db = evdb_open(&config);
EvdbEmbedder* emb = evdb_embedder_create("model.onnx", "vocab.txt", 2);

uint64_t chunk_id;
evdb_insert_text(db, emb, "Hello world", 1, 0, &chunk_id);

EvdbQueryHandle* q = evdb_query_text(db, emb, "hello", 5, 0);
printf("Top result: %s (score: %.3f)\n", evdb_result_text(q, 0), evdb_result_score(q, 0));
evdb_query_free(q);

evdb_close(db);
evdb_embedder_destroy(emb);
```

## Performance Targets

| Metric | Target | Platform |
|---|---|---|
| Query latency (10k chunks) | < 100ms | All |
| Embedding per sentence | < 20ms | All |
| Index build throughput | > 10 chunks/sec | All |
| Library size (stripped) | < 4 MB | Android arm64 |

## Building

```bash
# Desktop Debug
cmake --preset desktop-debug
cmake --build --preset desktop-debug

# Desktop Release
cmake --preset desktop-release
cmake --build --preset desktop-release

# Run tests
ctest --test-dir build/desktop-debug
```

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Android Integration](docs/android_integration.md)
- [iOS Integration](docs/ios_integration.md)
- [Python Integration](docs/python_integration.md)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
