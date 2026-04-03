# EdgeVDB SDK

> **Embeddable cross-platform vector database with HNSW ANN, hybrid retrieval, knowledge graph, relational object store, and CRDT-based sync — all in a single zero-dependency C++ library.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()
[![Platforms](https://img.shields.io/badge/platforms-Android%20%7C%20iOS%20%7C%20Desktop-lightgrey.svg)]()

## Features

| Feature | Description |
|---|---|
| **HNSW ANN Index** | 384-dim float32 vectors, M=16, ef=200/64 |
| **Hybrid Retrieval** | α·cosine + β·page_proximity + γ·keyword |
| **Knowledge Graph** | On-device NER → entity graph → multi-hop expansion |
| **Embedding Pipeline** | Optional — bring your own embeddings or use built-in |
| **Object Store** | Schema-less NoSQL with typed property indexing |
| **Relation Index** | Foreign key edges between objects |
| **CRDT Sync** | LWW vector clock sync across devices |
| **Stable C API** | `vectordb.h` — single header, ABI-stable |
| **Platform SDKs** | Android (JNI/Kotlin), iOS (Swift), Python (ctypes) |
| **Zero Dependencies** | ONNX Runtime is optional — core works everywhere |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       EdgeVDB SDK                        │
├──────────────┬──────────────┬──────────────┬─────────────┤
│  Python SDK  │  Android SDK │   C API      │  iOS SDK    │
│  (ctypes)    │  (JNI/Kotlin)│  (vectordb.h)│  (Swift)    │
├──────────────┴──────────────┴──────────────┴─────────────┤
│                   C++ Core Library                        │
├──────┬──────┬──────┬──────┬──────┬──────┬──────┬─────────┤
│ HNSW │Chunk │Object│ KG   │Hybrid│ Sync │Token │Embedder │
│Index │Store │Store │Engine│Ranker│Engine│Budget│(opt.)   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴─────────┘
```

## Quick Start

### Without ONNX — Pre-computed Embeddings (Recommended)

Use embeddings from **any provider** (OpenAI, Cohere, sentence-transformers, etc.):

#### Python
```python
from edgevdb import EdgeVDB

with EdgeVDB("./data") as db:
    embedding = your_provider.embed("Neural networks classify images")
    chunk_id = db.insert_chunk("Neural networks classify images", embedding,
                               doc_id=1, page_number=0)

    query_emb = your_provider.embed("image classification")
    results = db.query_vector(query_emb, query_text="image classification", top_k=5)
    for r in results:
        print(f"  [{r.score:.3f}] {r.text}")
```

#### Android (Kotlin)
```kotlin
val db = EdgeVDB.open(context)
val embedding: FloatArray = yourProvider.embed("Neural networks classify images")
val chunkId = db.insertChunk(embedding, "Neural networks classify images", docId = 1, pageNumber = 0)

db.queryVector(embedding, "image classification", topK = 5).use { results ->
    results.toList().forEach { println("${it.score}: ${it.text}") }
}
db.close()
```

#### C API
```c
EvdbConfig config;
evdb_default_config(&config);
config.storage_dir = "./data";

EvdbHandle* db = evdb_open(&config);

float embedding[384] = { /* from your provider */ };
uint64_t chunk_id;
evdb_insert_chunk(db, "Hello world", embedding, 1, 0, &chunk_id);

EvdbQueryHandle* q = evdb_query_vector(db, embedding, "hello", 5);
printf("Top: %s (%.3f)\n", evdb_result_text(q, 0), evdb_result_score(q, 0));
evdb_query_free(q);

evdb_close(db);
```

### With Built-in Embedder (ONNX optional)

```python
from edgevdb import EdgeVDB, Embedder

with EdgeVDB("./data") as db:
    embedder = Embedder("model.onnx", "vocab.txt")
    db.insert_text(embedder, "Hello world", doc_id=1, page_number=0)
    results = db.query_text(embedder, "hello", top_k=5)
    print(results.context_string)
```

## Building

```bash
# Desktop (Linux/macOS/WSL)
cmake --preset desktop-debug
cmake --build build/desktop-debug

# Desktop Release + Benchmarks
cmake --preset desktop-release
cmake --build build/desktop-release

# Android ARM64
export ANDROID_NDK="/path/to/ndk"
cmake --preset android-arm64
cmake --build build/android-arm64

# Run tests
./build/desktop-debug/tests/test_hnsw
./build/desktop-debug/tests/test_e2e_rag
```

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for full build instructions, integration guides, and publishing workflows.

## Performance

| Metric | Measured | Platform |
|---|---|---|
| Query latency (10k chunks) | < 100ms | Desktop |
| Index build throughput | ~2000 chunks/sec | Desktop Release |
| Library size (stripped) | < 4 MB | Android arm64 |

## Documentation

- [**Developer Guide**](DEVELOPER_GUIDE.md) — Building, integration, publishing
- [Architecture](docs/architecture.md) — System design and data flow
- [API Reference](docs/api_reference.md) — Complete C API documentation
- [Android Integration](docs/android_integration.md)
- [iOS Integration](docs/ios_integration.md)
- [Python Integration](docs/python_integration.md)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
