# EdgeVDB SDK — Comprehensive Developer Guide

> **EdgeVDB** is a zero-dependency, on-device vector database SDK built in C++17.  
> It provides HNSW-based similarity search, hybrid ranking, object storage, knowledge graphs, and CRDT sync — all in a single portable library.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Building the SDK](#3-building-the-sdk)
4. [Integration Guide — C/C++](#4-integration-guide--cc)
5. [Integration Guide — Python](#5-integration-guide--python)
6. [Integration Guide — Android (Kotlin/Java)](#6-integration-guide--android-kotlinjava)
7. [Integration Guide — iOS (Swift)](#7-integration-guide--ios-swift)
8. [With vs Without ONNX Runtime](#8-with-vs-without-onnx-runtime)
9. [Running Tests & Benchmarks](#9-running-tests--benchmarks)
10. [Publishing & Distribution](#10-publishing--distribution)
11. [API Reference](#11-api-reference)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    EdgeVDB SDK                          │
├──────────────┬──────────────┬──────────────┬────────────┤
│  Python SDK  │  Android SDK │   C API      │  iOS SDK   │
│  (ctypes)    │  (JNI/Kotlin)│  (vectordb.h)│  (Swift)   │
├──────────────┴──────────────┴──────────────┴────────────┤
│                   C++ Core Library                       │
├──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────┤
│ HNSW │Chunk │Object│ KG   │Hybrid│ Sync │Token │Embedder│
│Index │Store │Store │Engine│Ranker│Engine│Budget│(opt.)  │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴────────┘
```

### Key Design Principles
- **Zero required dependencies** — Pure C++17, no external libs needed
- **ONNX is optional** — Bring your own embeddings, or use the built-in embedder
- **Single header API** — Include `edgevdb/vectordb.h` and link one library
- **Cross-platform** — Linux, macOS, Windows, Android, iOS, Raspberry Pi

---

## 2. Prerequisites

### Minimum Requirements

| Tool | Version | Check Command |
|------|---------|---------------|
| CMake | ≥ 3.22 | `cmake --version` |
| Ninja | ≥ 1.10 | `ninja --version` |
| C++ compiler | C++17 support | `g++ --version` or `clang++ --version` |

### Platform-Specific

| Platform | Additional Requirement |
|----------|----------------------|
| Android | Android NDK ≥ r25 (tested with r27) |
| Python | Python ≥ 3.8 |
| iOS | Xcode ≥ 14, Swift ≥ 5.7 |

### Install Prerequisites (Ubuntu/WSL)

```bash
# System packages
sudo apt update && sudo apt install -y build-essential ninja-build python3 python3-pip

# Install modern CMake (if system version < 3.22)
pip3 install --user cmake
export PATH="$HOME/.local/bin:$PATH"  # Add to ~/.bashrc
```

### Install Android NDK (for Android builds only)

```bash
# Via Android SDK Manager
sdkmanager "ndk;27.1.12297006"

# Set environment variable
export ANDROID_NDK="$HOME/Android/Sdk/ndk/27.1.12297006"
```

---

## 3. Building the SDK

### Project Structure

```
edgevdb/
├── CMakeLists.txt          # Root build config
├── CMakePresets.json        # Build presets (debug/release/android)
├── core/
│   ├── CMakeLists.txt       # Core library build
│   ├── include/edgevdb/     # Public headers (vectordb.h, version.h)
│   ├── src/                 # C++ implementation
│   └── vendor/              # Vendored deps (nlohmann/json, doctest)
├── android/                 # Android SDK (Kotlin + JNI)
├── ios/                     # iOS SDK (Swift + C bridge)
├── python/                  # Python SDK (ctypes)
└── tests/                   # C++ tests and benchmarks
```

### 3.1 Desktop Build (Linux/macOS/WSL)

```bash
cd edgevdb

# Debug build (with tests)
cmake --preset desktop-debug
cmake --build build/desktop-debug

# Release build (with benchmarks, optimized)
cmake --preset desktop-release
cmake --build build/desktop-release
```

**Output artifacts:**

| File | Description |
|------|-------------|
| `build/*/core/libedgevdb_core.a` | Static library (link into your app) |
| `build/*/core/libedgevdb_shared.so` | Shared library — Linux (`.so`), macOS (`.dylib`), Windows (`.dll`) |
| `build/*/tests/test_*` | Test binaries |
| `build/desktop-release/tests/bench_*` | Benchmark binaries |

### 3.2 Android Build (Cross-Compilation)

```bash
# ARM64 (physical devices)
export ANDROID_NDK="/path/to/ndk"
cmake --preset android-arm64
cmake --build build/android-arm64

# x86_64 (emulators)
cmake --preset android-x86_64
cmake --build build/android-x86_64
```

**Output:** `build/android-*/core/libedgevdb_shared.so` — ready for `jniLibs/`

### 3.3 Custom Build (Manual CMake)

```bash
mkdir build && cd build
cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DEDGEVDB_BUILD_TESTS=OFF \
  -DEDGEVDB_BUILD_PYTHON=OFF \
  -DEDGEVDB_ENABLE_SYNC=ON
cmake --build .
```

### Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `EDGEVDB_BUILD_TESTS` | `ON` | Build test suite |
| `EDGEVDB_BUILD_PYTHON` | `OFF` | Build Python extension |
| `EDGEVDB_ENABLE_NEON` | `ON` | ARM NEON SIMD |
| `EDGEVDB_ENABLE_SSE2` | `ON` | x86 SSE2 SIMD |
| `EDGEVDB_ENABLE_SYNC` | `ON` | CRDT sync engine |

---

## 4. Integration Guide — C/C++

### 4.1 Linking

```cmake
# In your CMakeLists.txt
add_subdirectory(path/to/edgevdb)
target_link_libraries(your_app PRIVATE edgevdb_core)
target_include_directories(your_app PRIVATE path/to/edgevdb/core/include)
```

Or link the pre-built library:

```cmake
target_link_libraries(your_app PRIVATE /path/to/libedgevdb_core.a pthread)
```

### 4.2 Basic Usage (Without ONNX)

```c
#include "edgevdb/vectordb.h"
#include <stdio.h>

int main() {
    // 1. Configure
    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = "./my_database";

    // 2. Open database
    EvdbHandle* db = evdb_open(&config);
    if (!db) { printf("Failed to open\n"); return 1; }

    // 3. Insert with pre-computed embedding (NO ONNX NEEDED)
    float embedding[384] = {0};
    embedding[0] = 0.8f; embedding[1] = 0.3f; embedding[2] = 0.5f;
    // In production: get embeddings from your preferred provider

    uint64_t chunk_id;
    evdb_insert_chunk(db, "Machine learning finds patterns in data",
                      embedding, /*doc_id=*/1, /*page=*/0, &chunk_id);
    printf("Inserted chunk: %llu\n", (unsigned long long)chunk_id);

    // 4. Query with pre-computed embedding
    float query_emb[384] = {0};
    query_emb[0] = 0.7f; query_emb[1] = 0.4f;

    EvdbQueryHandle* results = evdb_query_vector(db, query_emb, "ML patterns", 5);
    int count = evdb_result_count(results);
    for (int i = 0; i < count; i++) {
        printf("  [%d] score=%.3f text=%s\n", i,
               evdb_result_score(results, i),
               evdb_result_text(results, i));
    }
    evdb_query_free(results);

    // 5. Object store
    uint64_t obj_id;
    evdb_object_put(db, "Document", "{\"title\":\"ML Intro\",\"pages\":42}", &obj_id);

    // 6. Relations
    evdb_relation_add(db, "has_chunk", obj_id, chunk_id);

    // 7. Save & close
    evdb_save(db);
    evdb_close(db);
    return 0;
}
```

### 4.3 Usage WITH ONNX (Auto-Embedding)

```c
#include "edgevdb/vectordb.h"

int main() {
    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = "./my_database";
    EvdbHandle* db = evdb_open(&config);

    // Create embedder (requires ONNX model files)
    EvdbEmbedder* embedder = evdb_embedder_create(
        "models/all-MiniLM-L6-v2.onnx",  // ONNX model
        "models/vocab.txt",               // WordPiece vocab
        2                                  // threads
    );

    // Insert — embedding computed automatically
    uint64_t chunk_id;
    evdb_insert_text(db, embedder, "AI transforms healthcare", 1, 0, &chunk_id);

    // Query — embedding computed automatically
    EvdbQueryHandle* results = evdb_query_text(db, embedder, "medical AI", 5, 0);
    // ... process results ...

    evdb_query_free(results);
    evdb_embedder_destroy(embedder);
    evdb_close(db);
    return 0;
}
```

---

## 5. Integration Guide — Python

### 5.1 Install

**Option A: From built shared library**
```bash
# Build the SDK first
cmake --preset desktop-release && cmake --build build/desktop-release

# Copy .so into the Python package (platform-specific)
# Linux:
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/lib/linux/

# Install in development mode
cd python && pip install -e .
```

**Option B: From published package (after publishing to PyPI)**
```bash
pip install edgevdb
```

### 5.2 Usage — Without ONNX (Recommended for Flexibility)

> **Note:** `insert_chunk(text, embedding, ...)` takes **text first**, then embedding.  
> This matches the C API order: `evdb_insert_chunk(handle, text, embedding, doc_id, page, out_id)`.

```python
from edgevdb import EdgeVDB

# Open database
db = EdgeVDB("/path/to/storage")

# Get embeddings from ANY provider you prefer:
# - OpenAI: openai.embeddings.create(model="text-embedding-3-small", input=text)
# - Sentence-Transformers: model.encode(text)
# - Cohere: co.embed(texts=[text])
# - Your own model

embedding = your_embedding_provider.embed("Machine learning finds patterns")
# Signature: insert_chunk(text, embedding, doc_id=0, page_number=0)
chunk_id = db.insert_chunk("Machine learning finds patterns", embedding,
                           doc_id=1, page_number=1)

# Query
query_emb = your_embedding_provider.embed("what is ML?")
results = db.query_vector(query_emb, query_text="what is ML?", top_k=5)

# QueryResults supports iteration — each item is a ChunkResult with attributes:
#   .chunk_id, .text, .score, .page_number
for r in results:
    print(f"  score={r.score:.3f}  text={r.text}")

# Object store (structured data alongside vectors)
doc_id = db.put_object("Document", {"title": "ML Intro", "author": "Alice"})
db.add_relation("has_chunk", doc_id, chunk_id)

db.save()
db.close()
```

### 5.3 Usage — With Built-in Embedder

```python
from edgevdb import EdgeVDB, Embedder

# Create embedder (uses ONNX model or built-in fallback)
embedder = Embedder("models/all-MiniLM-L6-v2.onnx", "models/vocab.txt")

with EdgeVDB("/path/to/storage") as db:
    # Auto-embed on insert
    chunk_id = db.insert_text(embedder, "Deep learning uses neural networks",
                              doc_id=1, page_number=3)

    # Auto-embed on query
    results = db.query_text(embedder, "neural network architecture", top_k=5)
    print(results.context_string)  # Pre-assembled RAG context
```

### 5.4 Context Manager & Error Handling

```python
from edgevdb import EdgeVDB, set_log_level, version

print(f"EdgeVDB v{version()}")
set_log_level(2)  # 0=off, 1=error, 2=info, 3=debug

try:
    with EdgeVDB("/tmp/my_db") as db:
        # Context manager auto-saves and closes
        db.insert_chunk("test", [0.0]*384, doc_id=0, page_number=0)

        obj = db.get_object(999)
        if obj is None:
            print("Object not found (returns None, doesn't throw)")

except RuntimeError as e:
    print(f"EdgeVDB error: {e}")
```

---

## 6. Integration Guide — Android (Kotlin/Java)

### 6.1 Add to Your Project

**Option A: Local AAR (from build)**

```kotlin
// settings.gradle.kts
include(":edgevdb")
project(":edgevdb").projectDir = file("path/to/edgevdb/android")

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

### 6.2 Usage — Without ONNX

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.ChunkResult

class MyRepository(private val context: Context) {

    private val db = EdgeVDB.open(context)

    fun indexDocument(text: String, embedding: FloatArray, docId: Int, page: Int): Long {
        // Embedding comes from your preferred provider:
        // - Android ML Kit
        // - TensorFlow Lite
        // - OpenAI API
        // - Any REST endpoint
        return db.insertChunk(embedding, text, docId, page)
    }

    fun search(queryEmbedding: FloatArray, query: String): List<ChunkResult> {
        // queryVector() returns QueryResults (AutoCloseable) — use .use{} to avoid handle leaks
        return db.queryVector(queryEmbedding, query, topK = 5).use { it.toList() }
    }

    fun storeMetadata(title: String, author: String): Long {
        return db.putObject("Document", mapOf("title" to title, "author" to author))
    }

    fun cleanup() {
        db.save()
        db.close()
    }
}
```

### 6.3 Usage — With Built-in Embedder

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.Embedder

// Place model.onnx and vocab.txt in app/src/main/assets/
val embedder = Embedder.fromAssets(context)
val db = EdgeVDB.open(context)

// Auto-embed
val chunkId = db.insertText(embedder, "Kotlin is great for Android", docId = 1, pageNumber = 0)

// queryText returns AutoCloseable QueryResults — always use .use{} to avoid handle leaks
db.queryText(embedder, "Android development", topK = 5).use { results ->
    results.toList().forEach { println("${it.score}: ${it.text}") }
}

embedder.destroy()
db.close()
```

---

## 7. Integration Guide — iOS (Swift)

### 7.1 Add to Your Project

**Option A: Swift Package Manager**

```swift
// Package.swift or Xcode → File → Add Package Dependencies
dependencies: [
    .package(url: "https://github.com/edgevdb/edgevdb.git", from: "1.0.0")
]
```

**Option B: Build XCFramework manually**

```bash
# Build universal framework (arm64 + x86_64 simulator)
cd ios
./build-xcframework.sh

# Output: build/EdgeVDB.xcframework
# Drag into Xcode → Frameworks, Libraries, and Embedded Content
```

### 7.2 Usage — Without ONNX

```swift
import EdgeVDB

class VectorStore {
    private let db: EdgeVDB

    init() throws {
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let storageDir = docsDir.appendingPathComponent("edgevdb").path
        db = try EdgeVDB(storageDir: storageDir)
    }

    func index(text: String, embedding: [Float], docId: UInt32, page: UInt32) throws -> UInt64 {
        // Embedding from any source: Core ML, REST API, etc.
        return try db.insertChunk(text: text, embedding: embedding, docId: docId, page: page)
    }

    func search(embedding: [Float], query: String, topK: Int = 5) throws -> [ChunkResult] {
        // queryVector returns QueryResults — call .toArray() to get [ChunkResult]
        let results = try db.queryVector(embedding: embedding, queryText: query, topK: topK)
        defer { results.close() }
        return results.toArray()
    }

    func close() {
        try? db.save()  // save() throws — use try? if errors are non-critical
        db.close()
    }
}
```

### 7.3 Usage — With Embedder

```swift
import EdgeVDB

let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
let storageDir = docsDir.appendingPathComponent("edgevdb").path

let embedder = try Embedder(
    modelPath: Bundle.main.path(forResource: "model", ofType: "onnx")!,
    vocabPath: Bundle.main.path(forResource: "vocab", ofType: "txt")!
)
let db = try EdgeVDB(storageDir: storageDir)

// Auto-embed
let chunkId = try db.insertText(embedder: embedder, text: "Swift is powerful", docId: 1, page: 0)

// queryText returns QueryResults — always close when done
let results = try db.queryText(embedder: embedder, query: "iOS development", topK: 5)
defer { results.close() }

for result in results.toArray() {
    print("\(result.score): \(result.text)")
}

embedder.destroy()
db.close()
```

---

## 8. With vs Without ONNX Runtime

### Decision Guide

| Scenario | Use ONNX? | Approach |
|----------|-----------|----------|
| **You already have embeddings** (OpenAI, Cohere, etc.) | ❌ No | `insert_chunk()` + `query_vector()` |
| **Offline-first mobile app** | ✅ Yes | Bundle `model.onnx` in assets |
| **Server-side with external ML** | ❌ No | Call embedding API, store in EdgeVDB |
| **Raspberry Pi / IoT** | ✅ Optional | Fallback embedder works without ORT |
| **Prototyping / testing** | ❌ No | Built-in hash embedder produces consistent vectors |
| **Production quality embeddings** | ✅ Yes | Link ONNX Runtime for MiniLM inference |

### Feature Availability Matrix

| Feature | Without ONNX | With ONNX |
|---------|:------------:|:---------:|
| `insert_chunk()` (pre-computed vectors) | ✅ | ✅ |
| `query_vector()` (pre-computed query) | ✅ | ✅ |
| `insert_text()` (auto-embed) | ⚠️ Hash fallback | ✅ Real embedding |
| `query_text()` (auto-embed) | ⚠️ Hash fallback | ✅ Real embedding |
| HNSW similarity search | ✅ | ✅ |
| Hybrid ranking (cosine + keyword + page) | ✅ | ✅ |
| Object store CRUD | ✅ | ✅ |
| Relation graph | ✅ | ✅ |
| Knowledge graph extraction | ✅ | ✅ |
| CRDT sync engine | ✅ | ✅ |
| Token budget assembly | ✅ | ✅ |
| Persistence (save/load) | ✅ | ✅ |

### Linking ONNX Runtime (Advanced)

To enable real model inference, link ONNX Runtime when building:

```bash
# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar xzf onnxruntime-linux-x64-1.20.0.tgz

# Build with ORT
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_ROOT=/path/to/onnxruntime-linux-x64-1.20.0
cmake --build build
```

Then modify `embedder.cpp` to call ORT APIs instead of `generateFallbackEmbedding()`.

---

## 9. Running Tests & Benchmarks

### After any build, verify:

```bash
# C++ unit tests (from project root, after desktop-debug build)
./build/desktop-debug/tests/test_hnsw           # HNSW index ops
./build/desktop-debug/tests/test_hybrid_ranker   # Ranking algorithm
./build/desktop-debug/tests/test_embedder        # Embedding pipeline
./build/desktop-debug/tests/test_object_store    # Object CRUD
./build/desktop-debug/tests/test_sync            # CRDT sync
./build/desktop-debug/tests/test_e2e_rag         # Full RAG pipeline

# Benchmarks (release build only)
./build/desktop-release/tests/bench_query        # Query latency at scale
./build/desktop-release/tests/bench_build        # Index build throughput

# Python SDK tests (uses unittest; pytest also works if installed)
cp build/desktop-debug/core/libedgevdb_shared.so python/edgevdb/lib/linux/
cd python && python3 -m unittest tests.test_edgevdb -v
# Or with pytest: cd python && pytest tests/ -v

# Android artifact verification
file build/android-arm64/core/libedgevdb_shared.so   # ELF 64-bit ARM aarch64
file build/android-x86_64/core/libedgevdb_shared.so  # ELF 64-bit x86-64
```

---

## 10. Publishing & Distribution

### 10.1 Python → PyPI

```bash
cd python

# 1. Copy the built shared library into the package (platform-specific)
# Linux:
cp ../build/desktop-release/core/libedgevdb_shared.so edgevdb/lib/linux/
# macOS:
# cp ../build/desktop-release/core/libedgevdb_shared.dylib edgevdb/lib/darwin/
# Windows (from PowerShell):
# copy ..\build\desktop-release\core\edgevdb_shared.dll edgevdb\lib\windows\

# 2. Build the wheel
pip install build
python3 -m build

# 3. Upload to PyPI
pip install twine
twine upload dist/*

# 4. Users install with:
#    pip install edgevdb
```

**For multi-platform wheels**, build on each OS (Linux, macOS, Windows) and use `cibuildwheel`:

```bash
pip install cibuildwheel
cibuildwheel --platform linux   # Builds manylinux wheels
```

**pyproject.toml** is already configured at `python/pyproject.toml`.

### 10.2 Android → Maven Central / GitHub Packages

```bash
# 1. Build the AAR
cd android
./gradlew assembleRelease

# Output: android/build/outputs/aar/edgevdb-release.aar
```

**Publish to Maven Central:**

```kotlin
// android/build.gradle.kts — add publishing plugin
plugins {
    id("com.android.library")
    id("maven-publish")
    id("signing")
}

publishing {
    publications {
        create<MavenPublication>("release") {
            groupId = "ai.edgevdb"
            artifactId = "edgevdb-android"
            version = "1.0.0"
            afterEvaluate {
                from(components["release"])
            }
        }
    }
    repositories {
        maven {
            url = uri("https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/")
            credentials {
                username = findProperty("ossrhUsername") as String?
                password = findProperty("ossrhPassword") as String?
            }
        }
    }
}
```

**Users add to their project:**

```kotlin
// build.gradle.kts
repositories {
    mavenCentral()
}
dependencies {
    implementation("ai.edgevdb:edgevdb-android:1.0.0")
}
```

### 10.3 C/C++ → System Package / CMake FetchContent

**Option A: CMake FetchContent (recommended)**

```cmake
# In consumer's CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
    edgevdb
    GIT_REPOSITORY https://github.com/edgevdb/edgevdb.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(edgevdb)
target_link_libraries(my_app PRIVATE edgevdb_core)
```

**Option B: Install system-wide**

```bash
cd edgevdb
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build
sudo cmake --install build

# Users link with CMake:
# find_package(edgevdb REQUIRED)
# target_link_libraries(app PRIVATE edgevdb_core)
```

**Option C: Conan package manager**

```python
# conanfile.py
from conan import ConanFile

class EdgeVDBConan(ConanFile):
    name = "edgevdb"
    version = "1.0.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"
    exports_sources = "CMakeLists.txt", "core/*", "CMakePresets.json"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
```

### 10.4 iOS → Swift Package / CocoaPods

**Option A: Swift Package Manager (recommended)**

```swift
// Package.swift
let package = Package(
    name: "EdgeVDB",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [
        .library(name: "EdgeVDB", targets: ["EdgeVDB"]),
    ],
    targets: [
        .target(
            name: "EdgeVDBCore",
            path: "core",
            sources: ["src"],
            publicHeadersPath: "include",
            cxxSettings: [.headerSearchPath("src"), .headerSearchPath("vendor")]
        ),
        .target(
            name: "EdgeVDB",
            dependencies: ["EdgeVDBCore"],
            path: "ios/Sources/EdgeVDB"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
```

**Option B: CocoaPods**

```ruby
# EdgeVDB.podspec
Pod::Spec.new do |s|
  s.name         = "EdgeVDB"
  s.version      = "1.0.0"
  s.summary      = "On-device vector database SDK"
  s.homepage     = "https://github.com/edgevdb/edgevdb"
  s.license      = "Apache-2.0"
  s.author       = "EdgeVDB Team"
  s.ios.deployment_target = "15.0"
  s.osx.deployment_target = "12.0"
  s.source       = { git: "https://github.com/edgevdb/edgevdb.git", tag: "v1.0.0" }

  # C++ core (source pod)
  s.subspec 'Core' do |core|
    core.source_files = "core/src/**/*.{cpp,hpp,h}", "core/include/**/*.h",
                        "core/vendor/**/*.{hpp,h}"
    core.public_header_files = "core/include/**/*.h"
    core.header_mappings_dir = "core/include"
    core.xcconfig = {
      'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
      'CLANG_CXX_LIBRARY' => 'libc++',
      'HEADER_SEARCH_PATHS' => '$(PODS_TARGET_SRCROOT)/core/src $(PODS_TARGET_SRCROOT)/core/vendor'
    }
  end

  # Swift wrapper
  s.subspec 'Swift' do |sw|
    sw.dependency 'EdgeVDB/Core'
    sw.source_files = "ios/Sources/EdgeVDB/**/*.swift"
  end

  s.default_subspecs = 'Core', 'Swift'
end
```

---

## 11. API Reference

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `EVDB_OK` | Success |
| 1 | `EVDB_ERR_NULL_HANDLE` | Null handle passed |
| 2 | `EVDB_ERR_IO` | I/O or internal error |
| 3 | `EVDB_ERR_OOM` | Out of memory |
| 4 | `EVDB_ERR_NOT_FOUND` | Record not found |
| 5 | `EVDB_ERR_INVALID_ARG` | Invalid argument |
| 6 | `EVDB_ERR_ONNX` | ONNX Runtime error |
| 7 | `EVDB_ERR_SYNC` | Sync error |

### C API Quick Reference

```
Lifecycle:     evdb_default_config, evdb_open, evdb_close, evdb_save
Embedding:     evdb_embedder_create, evdb_embedder_destroy, evdb_embed_text
Insert:        evdb_insert_text, evdb_insert_chunk, evdb_remove_chunk
Query:         evdb_query_text, evdb_query_vector, evdb_query_free
Results:       evdb_result_count, evdb_result_text, evdb_result_score,
               evdb_result_chunk_id, evdb_result_page, evdb_result_context_string
Objects:       evdb_object_put, evdb_object_get, evdb_object_remove, evdb_object_query
Relations:     evdb_relation_add, evdb_relation_get_targets
Sync:          evdb_sync_create, evdb_sync_destroy, evdb_sync_export_delta,
               evdb_sync_apply_delta, evdb_sync_export_to_file, evdb_sync_import_from_file
Utilities:     evdb_version_string, evdb_error_string, evdb_set_log_level
```

### Config Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hnsw_M` | 16 | HNSW max connections per node |
| `hnsw_ef_construction` | 200 | Build-time search width |
| `hnsw_ef_search` | 64 | Query-time search width |
| `ranker_alpha` | 0.70 | Cosine similarity weight |
| `ranker_beta` | 0.20 | Page proximity weight |
| `ranker_gamma` | 0.10 | Keyword match weight |
| `token_budget` | 3200 | Max tokens in RAG context |
| `embedding_threads` | 2 | ONNX inference threads |
| `enable_knowledge_graph` | 1 | Auto-extract entities |
| `enable_sync` | 0 | CRDT sync engine |

---

## Quick Start Summary

```
                    ┌─────────────────────────┐
                    │  Do you have your own   │
                    │  embedding provider?    │
                    └────────┬────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                   YES               NO
                    │                 │
           Use insert_chunk()    Use insert_text()
           Use query_vector()    Use query_text()
           (No ONNX needed)     (Needs embedder)
                    │                 │
              ┌─────┴─────┐    ┌─────┴──────┐
              │ Best for  │    │ Best for   │
              │ servers,  │    │ offline    │
              │ flexible  │    │ mobile,    │
              │ providers │    │ edge IoT   │
              └───────────┘    └────────────┘
```

**That's it!** Choose your path, build, and ship. EdgeVDB handles the rest.
