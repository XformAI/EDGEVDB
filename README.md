<div align="center">

# ⬡ EdgeVDB SDK

### On-Device Vector Database for the Edge

**Built by [XformAI](https://www.xformai.in/) — Transforming AI for the Real World**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-green.svg)]()
[![PyPI](https://img.shields.io/pypi/v/edgevdb.svg)](https://pypi.org/project/edgevdb/)
[![Android](https://img.shields.io/badge/android-GitHub%20Packages-brightgreen.svg)](https://github.com/XformAI/EDGEVDB/packages)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()
[![Platforms](https://img.shields.io/badge/platforms-Android%20%7C%20iOS%20%7C%20Desktop-lightgrey.svg)]()
[![Docs](https://img.shields.io/badge/docs-edgevdb.xformai.in-6366f1.svg)](https://xformai.github.io/EDGEVDB/)

[📖 Documentation](https://xformai.github.io/EDGEVDB/) · [🌐 XformAI Website](https://www.xformai.in/) · [🐛 Report Bug](https://github.com/XformAI/EDGEVDB/issues) · [💡 Request Feature](https://github.com/XformAI/EDGEVDB/issues)

</div>

---

> **Embeddable cross-platform vector database with HNSW ANN, hybrid retrieval, knowledge graph, relational object store, and CRDT-based sync — all in a single zero-dependency C++ library.**

## Why EdgeVDB?

- **Sub-millisecond queries** — 0.99ms avg at 10K vectors with 96.8% recall
- **Tiny footprint** — 428 KB desktop, 241 KB Android. No bloat.
- **Zero dependencies** — Pure C++17. ONNX Runtime is optional.
- **Truly cross-platform** — One codebase for Android, iOS, Python, Desktop, and IoT
- **Privacy-first** — All data stays on-device. No cloud, no telemetry, no API keys.
- **Built-in RAG stack** — HNSW + hybrid ranking + knowledge graph + token budgeting

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
    results.free()  # Always free native query handle
```

#### Android (Kotlin)
```kotlin
val db = EdgeVDB.open(context, dbPath = context.filesDir.resolve("edgevdb").path)
val embedding: FloatArray = yourProvider.embed("Neural networks classify images")
val chunk = DocumentChunk(text = "Neural networks classify images", docId = "doc-1", page = 0)
val chunkId = db.insertChunk(chunk, embedding)

val results = db.queryVector(embedding, topK = 5, queryText = "image classification")
results.forEach { println("${it.score}: ${it.text}") }
db.close()
```

#### iOS (Swift)
```swift
let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
let storageDir = docsDir.appendingPathComponent("edgevdb").path

let db = try EdgeVDB(storageDir: storageDir)
let embedding: [Float] = yourProvider.embed("Neural networks classify images")
let chunkId = try db.insertChunk(text: "Neural networks classify images",
                                 embedding: embedding, docId: 1, page: 0)

let results = try db.queryVector(embedding: embedding, queryText: "image classification", topK: 5)
defer { results.close() }
for r in results.toArray() {
    print("\(r.score): \(r.text)")
}

try? db.save()
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

## Installation

### Python (PyPI)

```bash
pip install edgevdb
```

```python
from edgevdb import EdgeVDB
with EdgeVDB("./data") as db:
    print("EdgeVDB ready!")
```

### Android (GitHub Packages)

```kotlin
// settings.gradle.kts — add the repository
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/XformAI/EDGEVDB")
        credentials {
            username = project.findProperty("gpr.user") as String?
            password = project.findProperty("gpr.token") as String?
        }
    }
}

// build.gradle.kts
dependencies {
    implementation("in.xformai:edgevdb-android:0.1.1")
}
```

> Requires a GitHub personal access token with `read:packages` scope. Add `gpr.user` and `gpr.token` to your `~/.gradle/gradle.properties`.

### Building from Source

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

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the full benchmark report with real measured data — query latency, insert throughput, recall accuracy, memory usage, library sizes, cross-platform results, and comparison with existing vector databases.

| Metric (10k chunks, 384-dim) | Measured |
|---|---:|
| Query latency (C++ native) | **0.99 ms avg** |
| Query latency (Python SDK) | **7.74 ms avg** |
| Insert throughput (C++ native) | **1,053 chunks/sec** |
| Recall@5 (HNSW) | **96.8%** |
| Recall@5 (Hybrid Ranker) | **95.8%** |
| Library size (stripped) | **428 KB** (desktop) / **241 KB** (Android arm64) |
| Memory (10k chunks) | **~23 MB** |

## Documentation

> **📖 Full interactive documentation: [xformai.github.io/EDGEVDB](https://xformai.github.io/EDGEVDB/)**

- [**Developer Guide**](DEVELOPER_GUIDE.md) — Building, integration, publishing
- [Architecture](docs/architecture.md) — System design and data flow
- [API Reference](docs/api_reference.md) — Complete C API documentation
- [Android Integration](docs/android_integration.md)
- [iOS Integration](docs/ios_integration.md)
- [Python Integration](docs/python_integration.md)

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read the [Developer Guide](DEVELOPER_GUIDE.md) for coding conventions and build setup.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

<div align="center">

**[XformAI](https://www.xformai.in/)** · Made with ❤️ for edge AI developers

[Website](https://www.xformai.in/) · [Documentation](https://xformai.github.io/EDGEVDB/) · [GitHub](https://github.com/XformAI/EDGEVDB)
