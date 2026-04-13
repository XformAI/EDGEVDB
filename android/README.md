# EdgeVDB Android SDK

> **Production-ready Android SDK for on-device vector database with HNSW ANN, hybrid retrieval, knowledge graph, and CRDT-based sync — all in a single zero-dependency native library.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![API](https://img.shields.io/badge/API-26%2B-brightgreen.svg)](https://android-arsenal.com/api?level=26)
[![Kotlin](https://img.shields.io/badge/Kotlin-1.9.22-purple.svg)](https://kotlinlang.org)
[![Min SDK](https://img.shields.io/badge/Min%20SDK-26-orange.svg)](https://developer.android.com/about/versions/android-4.0)

## Overview

The EdgeVDB Android SDK provides a complete on-device vector database solution for Android applications. It combines a high-performance C++ core with idiomatic Kotlin APIs, enabling:

- **HNSW ANN Index** — Fast approximate nearest neighbor search with 384-dim float32 vectors
- **Hybrid Retrieval** — α·cosine + β·page_proximity + γ·keyword ranking
- **Knowledge Graph** — On-device NER → entity graph → multi-hop expansion
- **ONNX Embeddings** — Optional built-in embedder using ONNX Runtime
- **Object Store** — Schema-less NoSQL with typed property indexing
- **Relation Index** — Foreign key edges between objects
- **CRDT Sync** — LWW vector clock sync across devices
- **Coroutines** — Full Kotlin coroutines support for async operations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Android Application                    │
│  (Jetpack Compose / ViewModels / Repository Layer)     │
├─────────────────────────────────────────────────────────┤
│                   EdgeVDB Android SDK                   │
│  ┌──────────────┬──────────────┬──────────────────────┐ │
│  │   Kotlin     │   ONNX       │   Coroutines        │ │
│  │   APIs       │   Runtime    │   (Dispatchers.IO)  │ │
│  └──────────────┴──────────────┴──────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                      JNI Layer                           │
│  (vectordb_jni.cpp - Kotlin ↔ C++ bridge)              │
├─────────────────────────────────────────────────────────┤
│                 EdgeVDB C++ Core                         │
│  (HNSW Index, Object Store, KG, Sync, Hybrid Ranker)    │
└─────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Vector Search** | HNSW ANN with M=16, ef=200/64, 384-dim vectors | ✅ Production |
| **Hybrid Ranking** | Cosine + page proximity + keyword weights | ✅ Production |
| **Knowledge Graph** | Entity extraction + graph traversal | ✅ Production |
| **Object Store** | JSON document store with indexing | ✅ Production |
| **Relations** | Typed edges between objects | ✅ Production |
| **CRDT Sync** | Last-writer-wins vector clock sync | ✅ Production |
| **ONNX Embedder** | all-MiniLM-L6-v2 with quantization | ✅ Production |
| **Text Chunker** | Sliding window chunking with overlap | ✅ Production |
| **RAG Engine** | End-to-end retrieval-augmented generation | ✅ Production |

## Quick Start

### 1. Add Dependency

**Option A: Local AAR (from build)**

```kotlin
// settings.gradle.kts
include(":edgevdb")
project(":edgevdb").projectDir = file("../edgevdb/android/sdk")

// app/build.gradle.kts
dependencies {
    implementation(project(":edgevdb"))
}
```

**Option B: Maven Central (after publishing)**

```kotlin
// build.gradle.kts
dependencies {
    implementation("ai.edgevdb:edgevdb-android:1.0.0")
}
```

### 2. Add Model Files

Place the ONNX model and vocabulary in your app's assets:

```
app/src/main/assets/
├── model.onnx                    # all-MiniLM-L6-v2 (quantized recommended)
└── vocab.txt                     # WordPiece vocabulary (30,522 tokens)
```

Download models from Hugging Face:
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Export to ONNX: `python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 models/`
- Quantize to INT8: Reduces size from ~90MB to ~22MB with minimal accuracy loss

### 3. Initialize Database

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.Embedder
import kotlinx.coroutines.runBlocking

class MyApplication : Application() {
    lateinit var db: EdgeVDB
    lateinit var embedder: Embedder

    override fun onCreate() {
        super.onCreate()

        // Initialize embedder from assets
        embedder = Embedder.fromAssets(
            context = this,
            modelAsset = "model.onnx",
            vocabAsset = "vocab.txt",
            threads = 2
        )

        // Open database in app-specific storage
        val dbPath = filesDir.absolutePath + "/edgevdb"
        db = EdgeVDB.open(
            context = this,
            dbPath = dbPath,
            dims = 384,
            enableKG = true,
            enableOnnx = false  // Use Kotlin ONNX pipeline
        )
    }
}
```

### 4. Basic Usage

#### Insert Document Chunks

```kotlin
// Using pre-computed embeddings (recommended for flexibility)
suspend fun indexDocument(text: String, docId: String) {
    val chunker = TextChunker(chunkSize = 200, chunkOverlap = 40)
    val chunks = chunker.chunk(text, docId)

    chunks.forEach { chunk ->
        val embedding = embedder.embed(chunk.text)
        val chunkId = db.insertChunk(chunk, embedding)
        Log.d("EdgeVDB", "Inserted chunk: $chunkId")
    }

    db.save()
}
```

#### Semantic Search

```kotlin
suspend fun search(query: String): List<ChunkResult> {
    val embedding = embedder.embed(query)
    val results = db.queryVector(embedding, topK = 5)

    results.forEach { result ->
        Log.d("EdgeVDB", "Score: ${result.score}, Text: ${result.text.take(50)}...")
    }

    return results
}
```

#### Object Store

```kotlin
// Store metadata
suspend fun storeDocumentMetadata(title: String, author: String) {
    val docId = db.putObject(
        collection = "Document",
        id = UUID.randomUUID().toString(),
        jsonBody = """{"title":"$title","author":"$author"}"""
    )
    db.save()
}

// Retrieve metadata
suspend fun getDocumentMetadata(docId: String): String? {
    return db.getObject("Document", docId)
}
```

#### Relations

```kotlin
suspend fun linkDocumentToChunk(docId: String, chunkId: Long) {
    db.addRelation(
        fromCollection = "Document",
        fromId = docId,
        relationType = "has_chunk",
        toCollection = "Chunk",
        toId = chunkId.toString()
    )
    db.save()
}
```

## Advanced Usage

### RAG Engine

The `RagEngine` provides a complete retrieval-augmented generation pipeline:

```kotlin
import ai.edgevdb.RagEngine
import ai.edgevdb.OnnxEmbeddingPipeline

// Initialize RAG engine
val pipeline = OnnxEmbeddingPipeline(
    context = this,
    modelAssetPath = "model.onnx",
    vocabAssetPath = "vocab.txt",
    useNnapi = true  // Enable NNAPI acceleration
)

val ragEngine = RagEngine(
    pipeline = pipeline,
    db = db,
    contextMaxTokens = 2048
)

// Ingest a full document
runBlocking {
    val chunkIds = ragEngine.ingestDocument(
        text = documentText,
        docId = "doc-123",
        chunkSize = 200,
        chunkOverlap = 40
    )

    // Query
    val result = ragEngine.query(
        query = "What is the main topic?",
        topK = 5
    )

    println("Context:\n${result.contextString}")
    println("Latency: ${result.latencyMs}ms")
}
```

### Configuration

Customize HNSW and ranking parameters:

```kotlin
val db = EdgeVDB.open(
    context = this,
    dbPath = dbPath,
    dims = 384,
    enableKG = true,
    enableOnnx = false
)

// Get statistics
runBlocking {
    val stats = db.stats()
    Log.d("EdgeVDB", "Database stats: $stats")
}
```

### Thread Safety

- All database operations are performed on `Dispatchers.IO`
- The native C++ core uses shared mutexes for thread safety
- Multiple coroutines can safely perform concurrent operations
- Always call `db.close()` when done to release native resources

## API Reference

### EdgeVDB

Main database entry point.

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `open(context, dbPath, dims, enableKG, enableOnnx, ...)` | Open/create database | `EdgeVDB` |
| `insertChunk(chunk, embedding)` | Insert chunk with pre-computed embedding | `Long` (chunk ID) |
| `insertText(text, meta)` | Insert text using C++ embedder | `Long` (chunk ID) |
| `queryVector(embedding, topK)` | Search with pre-computed embedding | `List<ChunkResult>` |
| `queryText(query, topK)` | Search using C++ embedder | `List<ChunkResult>` |
| `putObject(collection, id, jsonBody)` | Store JSON object | `Boolean` |
| `getObject(collection, id)` | Retrieve JSON object | `String?` |
| `addRelation(fromCol, fromId, relType, toCol, toId)` | Add typed edge | `Boolean` |
| `save()` | Flush to disk | `Unit` |
| `close()` | Release native resources | `Unit` |
| `stats()` | Get database statistics | `String` |

### Embedder

ONNX embedding wrapper.

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `fromAssets(context, modelAsset, vocabAsset, threads)` | Create from assets | `Embedder` |
| `fromFiles(modelPath, vocabPath, threads)` | Create from files | `Embedder` |
| `embed(text)` | Embed text to 384-dim vector | `FloatArray` |
| `destroy()` | Release native resources | `Unit` |

### RagEngine

Complete RAG pipeline orchestrator.

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `ingestChunk(chunk, precomputedEmbedding)` | Ingest single chunk | `Long` (chunk ID) |
| `ingestDocument(text, docId, chunkSize, chunkOverlap)` | Ingest full document | `List<Long>` (chunk IDs) |
| `query(query, topK)` | Full RAG query | `QueryResult` |

### Data Classes

#### ChunkResult

```kotlin
@Serializable
data class ChunkResult(
    val id: Long = 0L,
    val text: String = "",
    val score: Float = 0f,
    val meta: String = "",
    val docId: String = "",
    val page: Int = 0
)
```

#### DocumentChunk

```kotlin
data class DocumentChunk(
    val text: String,
    val docId: String,
    val chunkIndex: Int = 0,
    val page: Int = 0,
    val metadata: String = "{}"
)
```

#### QueryResult

```kotlin
data class QueryResult(
    val chunks: List<ChunkResult>,
    val contextString: String,
    val queryEmbedding: FloatArray,
    val latencyMs: Long
)
```

## Implementation Details

### Native Library

- **Language**: C++17
- **Build System**: CMake (via Android Gradle Plugin)
- **Architecture**: ARM NEON SIMD for vector operations
- **Library Size**: < 4 MB (stripped, arm64-v8a)
- **ABI**: Stable C API via JNI

### Kotlin Layer

- **Language**: Kotlin 1.9.22
- **Coroutines**: Full async support with `Dispatchers.IO`
- **Serialization**: kotlinx.serialization for JSON
- **Thread Safety**: All operations on IO dispatcher
- **Memory Management**: RAII patterns for native handles

### ONNX Runtime Integration

- **Version**: Latest from Maven Central
- **Model**: all-MiniLM-L6-v2 (sentence-transformers)
- **Dimensions**: 384
- **Quantization**: INT8 supported (recommended for mobile)
- **Acceleration**: NNAPI delegate support
- **Thread Count**: Configurable (default: 2)

## Building

### Prerequisites

- Android SDK 35
- Android NDK r27+
- Kotlin 1.9.22
- Gradle 8.0+

### Build SDK

```bash
cd android

# Build AAR
./gradlew :sdk:assembleRelease

# Output: sdk/build/outputs/aar/sdk-release.aar
```

### Build Demo App

```bash
cd android

# Build APK
./gradlew :demos:rag-demo:assembleDebug

# Install on device
./gradlew :demos:rag-demo:installDebug
```

### Run Tests

```bash
cd android

# Unit tests
./gradlew :sdk:test

# Instrumented tests
./gradlew :demos:rag-demo:connectedAndroidTest
```

## Demo Application

A complete Jetpack Compose demo is included in `demos/rag-demo/`:

- **UI**: Material 3 with Jetpack Compose
- **Architecture**: MVVM with Hilt dependency injection
- **Features**: Document ingestion, semantic search, RAG context display
- **Navigation**: Navigation Compose

Run the demo to see EdgeVDB in action:

```bash
cd android
./gradlew :demos:rag-demo:installDebug
```

## Troubleshooting

### Native Library Not Found

Ensure the native library is included in the AAR:

```kotlin
// sdk/build.gradle.kts
android {
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
}
```

### ONNX Model Loading Issues

- Verify model and vocab files are in `app/src/main/assets/`
- Check file sizes: model.onnx (~90MB or ~22MB quantized), vocab.txt (~230KB)
- Enable logging: `adb logcat | grep EdgeVDB`

### Out of Memory Errors

- Reduce chunk size when ingesting large documents
- Limit concurrent embedding operations
- Use quantized ONNX model to reduce memory footprint

## Contributing

We welcome contributions from the open-source community!

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `./gradlew test`
5. Build SDK: `./gradlew :sdk:assembleRelease`
6. Submit a pull request

### Code Style

- Kotlin: Follow [Kotlin coding conventions](https://kotlinlang.org/docs/coding-conventions.html)
- C++: Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use ktlint for Kotlin formatting
- Use clang-format for C++ formatting

### Testing

- Unit tests for all public APIs
- Instrumented tests for Android-specific code
- Performance benchmarks for critical paths
- Memory leak detection with LeakCanary

## License

Apache License 2.0 — see [../../LICENSE](../../LICENSE)

## Support

- **Documentation**: https://github.com/edgevdb/edgevdb/tree/main/docs
- **Issues**: https://github.com/edgevdb/edgevdb/issues
- **Discussions**: https://github.com/edgevdb/edgevdb/discussions
