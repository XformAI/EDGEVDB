# EdgeVDB Android RAG SDK

A production-quality Android SDK for on-device Retrieval-Augmented Generation (RAG) using EdgeVDB and ONNX Runtime.

## Overview

This project implements the EdgeVDB Android SDK as described in the `EdgeVDB_Android_RAG_SDK_Guide.md`. It includes:

- **edgevdb/android**: The Android library module containing:
  - JNI bridge to the EdgeVDB C++ core
  - ONNX Runtime-based embedding pipeline
  - Pure-Kotlin SimpleVectorDB fallback for testing
  - RAG engine orchestrating ingestion and query
  - Full Kotlin API with coroutine support

- **app**: A demo Android application showcasing:
  - Document ingestion with chunking
  - Semantic search with vector similarity
  - Jetpack Compose UI
  - Hilt dependency injection

## Prerequisites

- **Android Studio**: Koala | 2024.1.1 or later
- **NDK**: r25+ (r27 recommended)
- **CMake**: 3.22.1+
- **Python 3.8+**: For model export scripts
- **EdgeVDB C++ Core**: Located at `edgevdb/` relative to this project

## Quick Start

### 1. Prepare ONNX Model

The SDK requires a quantized all-MiniLM-L6-v2 ONNX model and vocabulary file.

```bash
# Option 1: Copy from existing EdgeVDB models directory
cp edgevdb/models/all-MiniLM-L6-v2-quantized.onnx edgevdb/android/src/main/assets/model.onnx
cp edgevdb/models/vocab.txt edgevdb/android/src/main/assets/vocab.txt

# Option 2: Export from Hugging Face
python3 -m venv .venv
source .venv/bin/activate
pip install optimum transformers torch onnxruntime
python3 -c "from optimum.exporters.onnx import main_export; main_export('sentence-transformers/all-MiniLM-L6-v2', './models', task='feature-extraction')"
python3 quantize_model.py --input models/model.onnx --output edgevdb/android/src/main/assets/model.onnx
# Copy vocab.txt from models/ to edgevdb/android/src/main/assets/
```

### 2. Configure NDK Path

Create or edit `local.properties`:

```properties
ndk.dir=/path/to/your/android-ndk
sdk.dir=/path/to/your/android-sdk
```

### 3. Build the Project

```bash
# Build debug APK
./gradlew :app:assembleDebug

# Install on connected device
./gradlew :app:installDebug

# Run tests
./gradlew :edgevdb:android:test :app:test
```

## Project Structure

```
EdgeVDB-Android/
├── edgevdb/android/             # SDK library module
│   ├── build.gradle.kts         # Gradle configuration
│   ├── CMakeLists.txt           # Native build config
│   ├── consumer-rules.pro       # ProGuard rules
│   ├── src/main/
│   │   ├── cpp/vectordb_jni.cpp # JNI bridge
│   │   ├── kotlin/ai/edgevdb/   # Kotlin API
│   │   │   ├── EdgeVDB.kt       # Main database class
│   │   │   ├── OnnxEmbeddingPipeline.kt
│   │   │   ├── RagEngine.kt
│   │   │   ├── SimpleVectorDB.kt
│   │   │   └── ...
│   │   └── assets/              # ONNX model & vocab
│   └── src/test/                # Unit tests
├── app/                         # Demo application
│   ├── build.gradle.kts
│   ├── src/main/
│   │   ├── kotlin/ai/edgevdb/demo/
│   │   │   ├── MainActivity.kt
│   │   │   ├── di/AppModule.kt
│   │   │   ├── data/
│   │   │   ├── viewmodel/
│   │   │   └── ui/
│   │   └── AndroidManifest.xml
│   └── src/test/                # App tests
├── gradle/
│   └── libs.versions.toml       # Dependency versions
├── settings.gradle.kts
├── build.gradle.kts
└── quantize_model.py            # Model quantization script
```

## Using the SDK

### Basic Usage

```kotlin
// Initialize EdgeVDB
val db = EdgeVDB.open(
    context = context,
    dbPath = "${context.filesDir.path}/mydb",
    dims = 384,
    enableKG = false
)

// Create embedding pipeline
val pipeline = OnnxEmbeddingPipeline(context)

// Create RAG engine
val ragEngine = RagEngine(pipeline = pipeline, db = db)

// Ingest a document
val chunkIds = ragEngine.ingestDocument(
    text = "Your document text here...",
    docId = "doc1"
)

// Query
val result = ragEngine.query("What is the document about?", topK = 5)
println(result.contextString)

// Cleanup
db.save()
db.close()
pipeline.close()
```

### Using SimpleVectorDB (Testing/CI)

```kotlin
// No ONNX model required - uses hash-based embeddings
val simpleDB = SimpleVectorDB(dimensions = 384)
val ragEngine = RagEngine(pipeline = simpleDB, simpleDB = simpleDB)
```

## Architecture

- **EdgeVDB C++ Core**: Provides HNSW vector search, object store, and knowledge graph
- **JNI Bridge**: Exposes C API to Kotlin via `vectordb_jni.cpp`
- **ONNX Runtime**: Executes all-MiniLM-L6-v2 model on-device
- **Kotlin SDK**: Provides coroutine-safe, type-safe API
- **RAG Pipeline**: Orchestrates chunking, embedding, indexing, and retrieval

## Performance

- **Embedding Latency**: ~35-55ms on Pixel 6 CPU (INT8 quantized)
- **Search Latency**: <5ms for 1,000 chunks, <100ms for 10,000 chunks
- **Model Size**: ~22 MB (INT8 quantized) vs ~90 MB (FP32)

## Testing

```bash
# Unit tests (JVM, no device required)
./gradlew :edgevdb-sdk:test
./gradlew :app:test

# With coverage
./gradlew :edgevdb-sdk:test jacocoTestReport
```

## Troubleshooting

See the troubleshooting section in `EdgeVDB_Android_RAG_SDK_Guide.md` for common issues.

## License

Apache License 2.0

## Related Documentation

- [EdgeVDB Android SDK Developer Guide](EdgeVDB_Android_RAG_SDK_Guide.md)
- [EdgeVDB Repository Documentation](EDGEVDB_REPOSITORY_DOCUMENTATION.md)
- [EdgeVDB Core](../edgevdb/README.md)
