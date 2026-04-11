# EDGEVDB Repository - Comprehensive Documentation

> **Complete documentation of the EDGEVDB repository structure, architecture, components, and implementation details.**

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Platform SDKs](#platform-sdks)
5. [Build System](#build-system)
6. [Configuration Files](#configuration-files)
7. [Source Code Analysis](#source-code-analysis)
8. [Testing & Benchmarks](#testing--benchmarks)
9. [Documentation](#documentation)
10. [Tools & Utilities](#tools--utilities)
11. [Models & Embeddings](#models--embeddings)
12. [Development Workflow](#development-workflow)

---

## Repository Overview

**EdgeVDB** is a zero-dependency, on-device vector database SDK built in C++17. It provides HNSW-based similarity search, hybrid ranking, object storage, knowledge graphs, and CRDT sync — all in a single portable library.

### Key Characteristics

- **Language**: C++17 (core), Kotlin (Android), Swift (iOS), Python (ctypes)
- **License**: Apache License 2.0
- **Version**: 1.0.0
- **Platforms**: Android, iOS, Desktop (Linux/macOS/Windows), Embedded
- **Dependencies**: Zero required dependencies (ONNX Runtime is optional)
- **Architecture**: Cross-platform with platform-specific SDKs

### Features

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

---

## Directory Structure

```
EDGEVDB/
├── .claude/                          # Claude AI configuration
│   ├── settings.json                 # Claude settings
│   └── skills/                       # Claude skills
│       ├── debug-issue.md
│       ├── explore-codebase.md
│       ├── refactor-safely.md
│       └── review-changes.md
├── .code-review-graph/               # Code review graph database
│   ├── .gitignore
│   └── graph.db                      # Knowledge graph database
├── .cursorrules                      # Cursor IDE rules
├── .git/                             # Git repository
├── .gitattributes                    # Git attributes
├── .mcp.json                         # MCP configuration
├── .opencode.json                    # OpenCode configuration
├── .venv2/                           # Python virtual environment
│   ├── bin/
│   ├── include/
│   ├── lib/
│   ├── lib64
│   └── pyvenv.cfg
├── .windsurfrules                    # Windsurf rules
├── AGENTS.md                         # Agent documentation
├── CLAUDE.md                         # Claude documentation
├── GEMINI.md                         # Gemini documentation
├── LICENSE                           # Apache License 2.0
├── edgevdb/                          # Main SDK directory
│   ├── .github/                      # GitHub configuration
│   ├── .gitignore                    # Git ignore patterns
│   ├── CMakeLists.txt                # Root CMake configuration
│   ├── CMakePresets.json             # CMake build presets
│   ├── DEVELOPER_GUIDE.md            # Comprehensive developer guide
│   ├── LICENSE                       # License file
│   ├── README.md                     # Main README
│   ├── android/                      # Android SDK
│   ├── build/                        # Build output directory
│   ├── core/                         # C++ core library
│   ├── demo_cli.py                   # Interactive CLI demo
│   ├── demo_quick.py                 # Quick demo script
│   ├── dist/                         # Distribution directory
│   ├── docs/                         # Documentation
│   ├── ios/                          # iOS SDK
│   ├── models/                       # Model files
│   ├── python/                       # Python SDK
│   ├── tests/                        # C++ tests
│   └── tools/                        # Utility tools
└── python=3.10/                      # Python environment
    ├── bin/
    ├── include/
    ├── lib/
    ├── lib64
    ├── pyvenv.cfg
    └── share/
```

---

## Core Components

### `edgevdb/core/` - C++ Core Library

The core library contains the main implementation of EdgeVDB in C++17.

#### Directory Structure

```
core/
├── CMakeLists.txt                    # Core library build configuration
├── include/
│   └── edgevdb/
│       ├── edgevdb.hpp              # C++ RAII wrapper (optional)
│       ├── version.h                # Version definitions
│       └── vectordb.h               # Stable public C API
├── src/                             # C++ implementation files
│   ├── chunk_store.cpp              # Chunk storage implementation
│   ├── chunk_store.hpp              # Chunk storage header
│   ├── embedder.cpp                 # Embedding pipeline implementation
│   ├── embedder.hpp                 # Embedding pipeline header
│   ├── hnsw_index.cpp               # HNSW index implementation
│   ├── hnsw_index.hpp               # HNSW index header
│   ├── hybrid_ranker.cpp            # Hybrid ranking implementation
│   ├── hybrid_ranker.hpp            # Hybrid ranking header
│   ├── kg_expander.cpp              # Knowledge graph expansion
│   ├── kg_expander.hpp              # Knowledge graph expansion header
│   ├── kg_extractor.cpp             # Knowledge graph extraction
│   ├── kg_extractor.hpp             # Knowledge graph extraction header
│   ├── knowledge_graph.cpp          # Knowledge graph implementation
│   ├── knowledge_graph.hpp          # Knowledge graph header
│   ├── log.hpp                      # Logging utilities
│   ├── object_store.cpp             # Object store implementation
│   ├── object_store.hpp             # Object store header
│   ├── page_index.cpp               # Page index implementation
│   ├── page_index.hpp               # Page index header
│   ├── query_engine.cpp             # Query engine implementation
│   ├── query_engine.hpp             # Query engine header
│   ├── relation_index.cpp           # Relation index implementation
│   ├── relation_index.hpp           # Relation index header
│   ├── schema.hpp                   # Data schema definitions
│   ├── sync_engine.cpp              # Sync engine implementation
│   ├── sync_engine.hpp              # Sync engine header
│   ├── sync_protocol.hpp            # Sync protocol definitions
│   ├── token_budget.cpp             # Token budget implementation
│   ├── token_budget.hpp             # Token budget header
│   ├── tokenizer.cpp                # Tokenizer implementation
│   ├── tokenizer.hpp                # Tokenizer header
│   ├── vectordb_c_api.cpp           # C API implementation
│   └── vectordb_impl.cpp            # Vector database implementation
└── vendor/                          # Vendored dependencies
    ├── README.md
    ├── doctest/
    │   └── doctest.h               # Testing framework
    ├── nlohmann/
    │   └── json.hpp                # JSON library
    └── onnxruntime/
        └── onnxruntime_c_api.h     # ONNX Runtime API
```

#### Key Source Files

**`vectordb.h`** - Stable Public C API (225 lines)
- Single header that external consumers need to include
- Provides stable ABI using opaque handle types
- Defines error codes, config struct, and all public functions
- Functions organized by category: Lifecycle, Embedding, Vector Store, Object Store, Relations, Sync, Utilities

**`hnsw_index.hpp`** - HNSW Index Header (166 lines)
- Implements HNSW (Hierarchical Navigable Small World) algorithm
- SIMD optimization for ARM NEON and x86 SSE2
- Cosine distance computation for L2-normalized vectors
- Multi-layer graph structure with configurable M, ef_construction, ef_search
- Thread-safe with shared_mutex for readers-writer locking

**Core Components Summary:**

| Component | File | Purpose |
|-----------|------|---------|
| Chunk Store | chunk_store.cpp/hpp | Flat file storage for text chunks |
| HNSW Index | hnsw_index.cpp/hpp | Approximate nearest neighbor search |
| Hybrid Ranker | hybrid_ranker.cpp/hpp | Multi-factor re-ranking (cosine + page + keyword) |
| Knowledge Graph | knowledge_graph.cpp/hpp | Entity extraction and graph storage |
| Object Store | object_store.cpp/hpp | Schema-less NoSQL storage |
| Relation Index | relation_index.cpp/hpp | Foreign key edges between objects |
| Sync Engine | sync_engine.cpp/hpp | CRDT-based synchronization |
| Query Engine | query_engine.cpp/hpp | Unified query orchestration |
| Embedder | embedder.cpp/hpp | ONNX-based text embedding |
| Tokenizer | tokenizer.cpp/hpp | WordPiece tokenization |
| Token Budget | token_budget.cpp/hpp | Context window management |

---

## Platform SDKs

### `edgevdb/android/` - Android SDK

Kotlin-based Android SDK with JNI bridge to C++ core.

#### Directory Structure

```
android/
├── .cxx/                            # CMake build artifacts
├── .gradle/                         # Gradle cache
├── CMakeLists.txt                    # Android CMake configuration
├── README.md                        # Android-specific README
├── app/                             # Application module
├── build/                           # Build output
├── build.gradle.kts                 # Gradle build configuration
├── edgevdb-sdk/                     # SDK module
├── local.properties                 # Local properties
└── src/
    ├── main/
    │   ├── assets/                  # Asset files
    │   │   ├── README.md
    │   │   ├── model.onnx          # ONNX model (90 MB)
    │   │   └── vocab.txt           # Vocabulary (262 KB)
    │   ├── cpp/                     # C++ JNI layer
    │   │   └── vectordb_jni.cpp   # JNI bridge (228 lines)
    │   └── kotlin/                  # Kotlin source
    │       └── ai/
    │           └── edgevdb/
    │               ├── ChunkResult.kt      # Query result data class
    │               ├── EdgeVDB.kt          # Main API class (156 lines)
    │               ├── Embedder.kt         # Embedder wrapper
    │               ├── ObjectRecord.kt     # Object record
    │               ├── SyncConfig.kt       # Sync configuration
    │               └── VectorDBManager.kt  # Database manager
    └── test/
        └── kotlin/
            └── ai/
                └── edgevdb/
                    └── EdgeVDBTest.kt     # Unit tests
```

#### Key Files

**`EdgeVDB.kt`** (156 lines)
- Main Kotlin API class
- Companion object with static initialization
- JNI native method declarations
- JSON serialization helpers (simplified)
- Methods: open, close, save, insertText, insertChunk, queryText, queryVector, putObject, getObject, addRelation

**`vectordb_jni.cpp`** (228 lines)
- JNI bridge from Kotlin to C API
- Implements all native methods declared in EdgeVDB.kt
- Handles jstring to std::string conversion
- Android logging integration
- Error handling with Java exceptions

**`build.gradle.kts`** (49 lines)
- Android library configuration
- Namespace: ai.edgevdb
- Compile SDK: 35, Min SDK: 26
- NDK ABI filters: arm64-v8a, x86_64
- CMake integration for C++ build
- Kotlin coroutines dependency

### `edgevdb/ios/` - iOS SDK

Swift-based iOS SDK with C bridge to C++ core.

#### Directory Structure

```
ios/
├── Sources/
│   ├── EdgeVDB/                      # Swift SDK
│   │   ├── EdgeVDB.swift           # Main API class (217 lines)
│   │   ├── Embedder.swift          # Embedder wrapper
│   │   ├── ObjectStore.swift       # Object store API
│   │   ├── SyncEngine.swift        # Sync engine API
│   │   └── VectorStore.swift       # Vector store API
│   ├── EdgeVDBC/                     # C bridge
│   │   └── bridge.h               # C bridge header
│   └── Tests/
│       └── EdgeVDBTests.swift      # Unit tests
└── build-xcframework.sh             # XCFramework build script
```

#### Key Files

**`EdgeVDB.swift`** (217 lines)
- Main Swift API class
- EdgeVDBConfig struct for configuration
- Embedder class for text embedding
- QueryResults class for query results
- ChunkResult struct for individual results
- EdgeVDBError enum for error handling
- Methods: init, close, save, insertText, insertChunk, removeChunk, queryText, putObject, getObject, removeObject, addRelation

**`bridge.h`** (94 bytes)
- C bridge header for Swift-C++ interop
- Exposes C API functions to Swift

**`build-xcframework.sh`** (1310 bytes)
- Shell script to build universal XCFramework
- Supports arm64 (physical devices) and x86_64 (simulator)

### `edgevdb/python/` - Python SDK

Python SDK using ctypes to interface with C++ shared library.

#### Directory Structure

```
python/
├── edgevdb/                          # Python package
│   ├── __init__.py                  # Main module (398 lines)
│   ├── __pycache__/                 # Python cache
│   ├── embedder.py                  # Embedder module (134 bytes)
│   ├── libedgevdb_shared.so         # Shared library (8.9 MB)
│   ├── object_store.py              # Object store module (782 bytes)
│   ├── sync.py                      # Sync module (830 bytes)
│   └── vectordb.py                  # Vector database module (290 bytes)
├── examples/                        # Example scripts
│   ├── object_sync.py              # Object sync example
│   └── pdf_rag.py                  # PDF RAG example
├── pyproject.toml                   # Python package configuration
└── tests/                           # Python tests
    ├── __pycache__/
    └── test_edgevdb.py             # Unit tests
```

#### Key Files

**`__init__.py`** (398 lines)
- Main Python module with ctypes wrapper
- Library loading with platform-specific search paths
- EvdbConfig ctypes structure
- EdgeVDB class with context manager support
- Embedder class for text embedding
- QueryResults class with lazy access
- ChunkResult data class
- Functions: version, set_log_level

**`pyproject.toml`** (38 lines)
- Python package configuration
- Name: edgevdb, Version: 1.0.0
- Requires Python 3.8+
- Build system: setuptools
- Classifiers for PyPI

---

## Build System

### CMake Configuration

**`CMakeLists.txt`** (root, 20 lines)
- Minimum CMake version: 3.22
- C++17 standard
- Build options: tests, Python, NEON, SSE2, sync
- Subdirectories: core, tests

**`CMakePresets.json`** (55 lines)
- Predefined build configurations
- desktop-debug: Debug build with Ninja
- desktop-release: Release build with -O3 -DNDEBUG
- android-arm64: ARM64 Android build
- android-x86_64: x86_64 Android build (emulators)

**`core/CMakeLists.txt`** (104 lines)
- Vendor library (INTERFACE)
- Core library sources (26 files)
- Static library: edgevdb_core
- Shared library: edgevdb_shared
- Platform-specific flags (Android, ARM64, x86_64)
- SIMD options (NEON, SSE2)
- Install rules for headers and libraries

### Build Presets

| Preset | Platform | Build Type | Output |
|--------|----------|------------|--------|
| desktop-debug | Linux/macOS/WSL | Debug | build/desktop-debug/ |
| desktop-release | Linux/macOS/WSL | Release | build/desktop-release/ |
| android-arm64 | Android (ARM64) | Release | build/android-arm64/ |
| android-x86_64 | Android (x86_64) | Release | build/android-x86_64/ |

### Build Commands

```bash
# Desktop Debug
cmake --preset desktop-debug
cmake --build build/desktop-debug

# Desktop Release
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

---

## Configuration Files

### Root Configuration

**`.gitignore`** (57 lines)
- Build artifacts: build/, cmake-build-*/, *.o, *.a, *.so, *.dll
- IDE: .idea/, .vscode/, *.swp, .DS_Store
- CMake: CMakeFiles/, CMakeCache.txt, Makefile
- Models: models/*.onnx, models/*.bin
- Python: __pycache__/, *.pyc, *.egg-info/, dist/
- Android: android/.gradle/, android/build/, *.apk
- iOS: ios/build/, ios/DerivedData/, Pods/

**`.gitattributes`** (68 bytes)
- Git LFS configuration for large files

**`.mcp.json`** (160 bytes)
- MCP (Model Context Protocol) configuration

**`.opencode.json`** (177 bytes)
- OpenCode configuration

**`.cursorrules`** (1755 bytes)
- Cursor IDE rules and guidelines

**`.windsurfrules`** (1755 bytes)
- Windsurf IDE rules and guidelines

### Agent Documentation

**`AGENTS.md`** (1755 bytes)
- Agent-specific guidelines and rules

**`CLAUDE.md`** (1755 bytes)
- Claude AI agent documentation

**`GEMINI.md`** (1755 bytes)
- Gemini AI agent documentation

### Claude Configuration

**`.claude/settings.json`** (447 bytes)
- Claude AI settings

**`.claude/skills/`**
- `debug-issue.md` (1087 bytes) - Debugging workflow
- `explore-codebase.md` (1205 bytes) - Codebase exploration
- `refactor-safely.md` (1197 bytes) - Safe refactoring
- `review-changes.md` (1067 bytes) - Code review workflow

---

## Source Code Analysis

### Core C++ Implementation

#### Data Flow

**Insert (with pre-computed embedding):**
1. Caller provides text + 384-dim float32 embedding
2. ChunkStore.put() → assigns ID and timestamp
3. HNSWIndex.insert() → builds graph connections
4. PageIndex.insert() → maps chunk to doc/page
5. KGExtractor.extract() → NER entities
6. KnowledgeGraph.addChunkEntities() → entity-chunk mapping

**Insert (with auto-embedding):**
1. Text → WordPiece tokenizer → token IDs
2. Token IDs → ONNX Runtime inference → 384-dim float32 embedding
3. L2 normalize embedding
4. Same steps as pre-computed path (2–6)

**Query:**
1. Query text → WordPiece tokenize → ONNX inference → 384-dim embedding
2. HNSW KNN search with over-fetch (3× top_k)
3. HybridRanker re-ranks: α·cosine + β·page_proximity + γ·keyword
4. Optional: KG expansion adds related chunks via entity graph
5. TokenBudget trims to fit LLM context window
6. Returns QueryResult[] and assembled context string

#### Binary File Formats

All stores use a consistent format:
- 8-byte magic (store-specific)
- 4-byte version (uint32)
- 8-byte record count (uint64)
- N × record (struct, trivially-copyable)
- 4-byte CRC32 trailer

#### Concurrency

All stores use `std::shared_mutex` for readers-writer locking:
- Multiple concurrent readers
- Exclusive writers
- Thread-safe from JNI/Swift/Python

### Platform-Specific Implementations

#### Android JNI Layer

**JNI Naming Convention:** `Java_ai_edgevdb_EdgeVDB_<methodName>`

Key JNI Functions:
- `nativeOpen` - Opens database with configuration
- `nativeClose` - Closes database handle
- `nativeSave` - Saves data to disk
- `nativeInsertText` - Inserts text with auto-embedding
- `nativeInsertChunk` - Inserts chunk with pre-computed embedding
- `nativeQueryText` - Queries with auto-embedding
- `nativeQueryVector` - Queries with pre-computed embedding
- `nativeObjectPut` - Stores object in object store
- `nativeObjectGet` - Retrieves object from object store
- `nativeRelationAdd` - Adds relation between objects

#### iOS Swift Layer

Swift classes provide idiomatic Swift APIs:
- `EdgeVDB` - Main database class
- `EdgeVDBConfig` - Configuration struct
- `Embedder` - Text embedder
- `QueryResults` - Query results with array access
- `ChunkResult` - Individual result
- `EdgeVDBError` - Error enum with Swift Error conformance

#### Python ctypes Layer

Python module uses ctypes for FFI:
- `_Lib` - Lazy-loaded library singleton
- `EvdbConfig` - ctypes.Structure for configuration
- `EdgeVDB` - Main database class with context manager
- `Embedder` - Text embedder
- `QueryResults` - Query results with iteration
- `ChunkResult` - Result data class

---

## Testing & Benchmarks

### C++ Tests

**`tests/`** Directory Structure

```
tests/
├── CMakeLists.txt                    # Test build configuration
├── benchmarks/
│   ├── bench_build_time.cpp         # Index build benchmark
│   └── bench_query_latency.cpp      # Query latency benchmark
├── test_e2e_rag.cpp                 # End-to-end RAG test
├── test_embedder.cpp                # Embedder test
├── test_hnsw.cpp                    # HNSW index test
├── test_hybrid_ranker.cpp           # Hybrid ranker test
├── test_object_store.cpp            # Object store test
├── test_sync.cpp                    # Sync engine test
```

#### Test Files

**`test_hnsw.cpp`** (3659 bytes)
- HNSW index operations test
- Insert, search, remove operations
- Accuracy and performance validation

**`test_hybrid_ranker.cpp`** (1584 bytes)
- Hybrid ranking algorithm test
- Multi-factor scoring validation

**`test_embedder.cpp`** (1408 bytes)
- Embedding pipeline test
- Tokenization and embedding generation

**`test_object_store.cpp`** (2008 bytes)
- Object CRUD operations test
- JSON serialization validation

**`test_sync.cpp`** (1840 bytes)
- CRDT sync engine test
- Delta export/import validation

**`test_e2e_rag.cpp`** (3475 bytes)
- Full RAG pipeline test
- Integration test for all components

#### Benchmarks

**`bench_query_latency.cpp`** (4113 bytes)
- Measures query latency at scale
- Tests with various dataset sizes
- Performance target: < 100ms for 10k chunks

**`bench_build_time.cpp`** (1659 bytes)
- Measures index build throughput
- Tests HNSW construction performance

### Python Tests

**`python/tests/test_edgevdb.py`**
- Python SDK unit tests
- Uses unittest framework
- Tests all Python API methods

### Running Tests

```bash
# C++ unit tests
./build/desktop-debug/tests/test_hnsw
./build/desktop-debug/tests/test_hybrid_ranker
./build/desktop-debug/tests/test_embedder
./build/desktop-debug/tests/test_object_store
./build/desktop-debug/tests/test_sync
./build/desktop-debug/tests/test_e2e_rag

# Benchmarks (release build only)
./build/desktop-release/tests/bench_query
./build/desktop-release/tests/bench_build

# Python tests
cp build/desktop-debug/core/libedgevdb_shared.so python/edgevdb/
cd python && python3 -m unittest tests.test_edgevdb -v
```

---

## Documentation

### Main Documentation Files

**`README.md`** (170 lines)
- Project overview and features
- Quick start guides for all platforms
- Building instructions
- Performance targets
- Links to detailed documentation

**`DEVELOPER_GUIDE.md`** (915 lines)
- Comprehensive developer guide
- Architecture overview
- Prerequisites for each platform
- Build instructions (desktop, Android, custom)
- Integration guides for C/C++, Python, Android, iOS
- ONNX Runtime decision guide
- Test and benchmark instructions
- Publishing and distribution workflows
- API reference summary

### Platform-Specific Documentation

**`docs/architecture.md`** (88 lines)
- System overview
- Component architecture diagram
- Data flow for insert and query
- Binary file formats
- Concurrency model
- Platform support matrix

**`docs/api_reference.md`** (144 lines)
- Complete C API reference
- Lifecycle functions
- Embedding functions
- Vector store operations
- Object store operations
- Relation operations
- Sync operations
- Utility functions
- Error codes
- Config defaults

**`docs/android_integration.md`** (135 lines)
- Android prerequisites
- Setup options (local module, Maven Central)
- Usage examples (with and without ONNX)
- ViewModel example
- ProGuard rules
- Build presets

**`docs/ios_integration.md`** (105 lines)
- iOS prerequisites
- Setup options (SPM, XCFramework, CocoaPods)
- Usage examples (with and without ONNX)
- Object store and relations
- Podspec reference

**`docs/python_integration.md`** (131 lines)
- Python prerequisites
- Building the library
- Setup instructions
- Usage examples (with and without ONNX)
- Object store and relations
- Utility functions
- API summary
- Running tests

---

## Tools & Utilities

### `tools/` Directory

```
tools/
├── convert_onnx.py                  # ONNX model conversion
├── download_model.py                # Model download script
└── validate_index.cpp               # Index validation tool
```

#### Tool Files

**`convert_onnx.py`** (2661 bytes)
- Converts models to ONNX format
- Handles quantization options
- Supports various input formats

**`download_model.py`** (1821 bytes)
- Downloads pre-trained models
- Fetches from Hugging Face
- Handles vocabulary files

**`validate_index.cpp`** (3179 bytes)
- Validates HNSW index integrity
- Checks graph structure
- Verifies embedding consistency

### Demo Scripts

**`demo_cli.py`** (295 lines)
- Interactive CLI demo
- Commands: add, query, list, object, relate, stats, save, clear, help, quit
- Color-coded terminal output
- Supports both real ONNX and hash fallback embeddings
- Shows embedding statistics and similarity scores

**`demo_quick.py`** (109 lines)
- Quick demonstration script
- Inserts sample documents
- Performs queries
- Demonstrates object store and relations
- Simple output without interactivity

---

## Models & Embeddings

### `models/` Directory

```
models/
├── .gitkeep                          # Git keep file
├── README.md                         # Models documentation
├── download_models.sh                # Download script
├── model.onnx                        # ONNX model (90 MB)
└── vocab.txt                         # Vocabulary (231 KB)
```

### Model Documentation

**`README.md`** (40 lines)
- Required files description
- Download instructions
- Manual download from Hugging Face
- ONNX export command
- Quantization instructions for mobile
- Size comparison: full (~90 MB) vs quantized (~22 MB)

### Required Files

| File | Description | Size (approx) |
|------|-------------|---------------|
| `all-MiniLM-L6-v2.onnx` | Sentence-BERT embedding model | ~90 MB |
| `all-MiniLM-L6-v2-quantized.onnx` | INT8 quantized model | ~22 MB |
| `vocab.txt` | WordPiece vocabulary (30,522 tokens) | ~230 KB |

### Model Download

```bash
cd tools
python download_model.py
```

Or manual download from Hugging Face:
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Export: `python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 models/`

### Quantization

For mobile deployment, quantize to INT8:
```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "all-MiniLM-L6-v2.onnx",
    "all-MiniLM-L6-v2-quantized.onnx",
    weight_type=QuantType.QInt8
)
```

The quantized model reduces size by ~4× with minimal accuracy loss (~0.1% on STS benchmarks).

---

## Development Workflow

### Prerequisites

| Tool | Version | Check Command |
|------|---------|---------------|
| CMake | ≥ 3.22 | `cmake --version` |
| Ninja | ≥ 1.10 | `ninja --version` |
| C++ compiler | C++17 support | `g++ --version` or `clang++ --version` |

### Platform-Specific Requirements

| Platform | Additional Requirement |
|----------|----------------------|
| Android | Android NDK ≥ r25 (tested with r27) |
| Python | Python ≥ 3.8 |
| iOS | Xcode ≥ 14, Swift ≥ 5.7 |

### Building the SDK

#### Desktop Build

```bash
cd edgevdb

# Debug build (with tests)
cmake --preset desktop-debug
cmake --build build/desktop-debug

# Release build (with benchmarks)
cmake --preset desktop-release
cmake --build build/desktop-release
```

**Output artifacts:**

| File | Description |
|------|-------------|
| `build/*/core/libedgevdb_core.a` | Static library |
| `build/*/core/libedgevdb_shared.so` | Shared library |
| `build/*/tests/test_*` | Test binaries |
| `build/desktop-release/tests/bench_*` | Benchmark binaries |

#### Android Build

```bash
# ARM64 (physical devices)
export ANDROID_NDK="/path/to/ndk"
cmake --preset android-arm64
cmake --build build/android-arm64

# x86_64 (emulators)
cmake --preset android-x86_64
cmake --build build/android-x86_64
```

**Output:** `build/android-*/core/libedgevdb_shared.so`

### Integration Examples

#### C/C++ Integration

```cmake
add_subdirectory(path/to/edgevdb)
target_link_libraries(your_app PRIVATE edgevdb_core)
target_include_directories(your_app PRIVATE path/to/edgevdb/core/include)
```

#### Python Integration

```bash
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/
cd python && pip install -e .
```

#### Android Integration

```kotlin
// settings.gradle.kts
include(":edgevdb")
project(":edgevdb").projectDir = file("path/to/edgevdb/android")

// app/build.gradle.kts
dependencies {
    implementation(project(":edgevdb"))
}
```

#### iOS Integration

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/edgevdb/edgevdb.git", from: "1.0.0")
]
```

### Publishing

#### Python → PyPI

```bash
cd python
cp ../build/desktop-release/core/libedgevdb_shared.so edgevdb/
pip install build
python3 -m build
pip install twine
twine upload dist/*
```

#### Android → Maven Central

```bash
cd android
./gradlew assembleRelease
# Output: android/build/outputs/aar/edgevdb-release.aar
```

#### C/C++ → System Package

```bash
cd edgevdb
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

### Performance Targets

| Metric | Target | Platform |
|---|---|---|
| Query latency (10k chunks) | < 100ms | Desktop |
| Library size (stripped) | < 4 MB | Android arm64 |

---

## Summary

EDGEVDB is a comprehensive, cross-platform vector database SDK designed for on-device RAG applications. The repository is well-organized with:

- **Core C++ library** with zero required dependencies
- **Platform SDKs** for Android (Kotlin/JNI), iOS (Swift), and Python (ctypes)
- **Comprehensive documentation** including architecture, API reference, and integration guides
- **Build system** using CMake with presets for different platforms
- **Testing infrastructure** with unit tests and benchmarks
- **Tools** for model conversion and download
- **Demo scripts** for quick start and interactive exploration

The architecture follows a layered approach with a stable C API at the core, platform-specific wrappers on top, and optional ONNX Runtime integration for on-device embedding generation. The codebase is production-ready with proper error handling, concurrency support, and cross-platform compatibility.

---

**Documentation Version:** 1.0.0  
**Last Updated:** April 2026  
**Repository:** XformAI/EDGEVDB  
**License:** Apache License 2.0
