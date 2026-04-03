# EdgeVDB SDK — Complete Agentic IDE Build Guide
## From Scratch: Cross-Platform Vector Database SDK with Embeddings, Relational DB & Sync

**SDK Name:** `EdgeVDB`  
**Target Platforms:** Android (arm64-v8a, x86_64), iOS (arm64), Desktop (macOS/Linux/Windows x86-64)  
**Language:** C++17 core · Kotlin (Android) · Swift (iOS) · Python (Desktop/Pi)  
**License:** Apache 2.0  
**Version:** 1.0.0

---

## TABLE OF CONTENTS

1. [Project Vision & Scope](#1-project-vision--scope)
2. [Repository Structure](#2-repository-structure)
3. [Phase 0 — Toolchain & Scaffolding](#phase-0--toolchain--scaffolding)
4. [Phase 1 — C++ Core: Schema & ChunkStore](#phase-1--c-core-schema--chunkstore)
5. [Phase 2 — HNSW Vector Index](#phase-2--hnsw-vector-index)
6. [Phase 3 — Hybrid Ranker & Page Proximity](#phase-3--hybrid-ranker--page-proximity)
7. [Phase 4 — Knowledge Graph Layer](#phase-4--knowledge-graph-layer)
8. [Phase 5 — Embedding Pipeline (Text → Vector)](#phase-5--embedding-pipeline-text--vector)
9. [Phase 6 — Relational Object Store](#phase-6--relational-object-store)
10. [Phase 7 — Data Sync Engine](#phase-7--data-sync-engine)
11. [Phase 8 — Public C API Facade](#phase-8--public-c-api-facade)
12. [Phase 9 — Android SDK (JNI + Kotlin)](#phase-9--android-sdk-jni--kotlin)
13. [Phase 10 — iOS SDK (C API + Swift)](#phase-10--ios-sdk-c-api--swift)
14. [Phase 11 — Desktop/Python SDK](#phase-11--desktoppython-sdk)
15. [Phase 12 — CMake Unified Build System](#phase-12--cmake-unified-build-system)
16. [Phase 13 — Testing & Benchmarks](#phase-13--testing--benchmarks)
17. [Phase 14 — Documentation & Packaging](#phase-14--documentation--packaging)
18. [Why Relational DB & Sync Were Missing — Design Rationale](#why-relational-db--sync-were-missing--design-rationale)
19. [Full Feature Matrix](#full-feature-matrix)

---

## 1. Project Vision & Scope

### What You Are Building

EdgeVDB SDK is a **single embeddable C++ library** that provides every layer needed for on-device RAG and AI-powered applications:

```
┌─────────────────────────────────────────────────────────────┐
│                     EdgeVDB SDK                              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Vector DB   │  │  Relational  │  │   Data Sync       │  │
│  │  (HNSW ANN)  │  │  Object DB   │  │   Engine          │  │
│  │  Hybrid Rank │  │  (No-SQL ORM)│  │   (CRDTs)         │  │
│  │  KG Expand   │  │              │  │                   │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         Embedding Pipeline (Text → Vector)           │    │
│  │  Tokenizer → ONNX MiniLM → L2-norm 384-dim vectors   │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                 Pure C Public API                    │    │
│  │              (vectordb.h — stable ABI)               │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │ Android    │  │ iOS Swift  │  │ Python (Desktop / Pi)  │  │
│  │ JNI/Kotlin │  │ Wrapper    │  │ ctypes harness         │  │
│  └────────────┘  └────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Core Capabilities

| Capability | Detail |
|---|---|
| Vector storage | 384-dim float32 embeddings, L2-normalised |
| ANN search | HNSW (M=16, ef_construction=200, ef_search=64) |
| Hybrid retrieval | α·cosine + β·page_proximity + γ·keyword_overlap |
| Document structure | Page/section metadata preserved as ranking signal |
| Knowledge graph | On-device NER → entity graph → multi-hop expansion |
| Embedding pipeline | Text input → WordPiece tokenize → ONNX MiniLM → vector |
| Relational store | Object-oriented NoSQL with typed properties & relations |
| Data sync | CRDT-based bi-directional sync across devices (optional) |
| Persistence | Binary-format flat files, mmap-compatible |
| Platform bindings | Android JNI, iOS C API, Python ctypes |
| Token budgeting | RAG context assembly with configurable token limit |

---

## 2. Repository Structure

Tell your agentic IDE to create this exact directory tree before writing any code:

```
edgevdb/
├── CMakeLists.txt                     # Root CMake (targets all platforms)
├── CMakePresets.json                  # Preset configs per platform
├── README.md
├── LICENSE
├── CHANGELOG.md
│
├── core/                              # ★ C++17 library source
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── edgevdb/
│   │       ├── vectordb.h             # Stable C public API
│   │       ├── edgevdb.hpp            # C++ RAII header (optional)
│   │       └── version.h              # Version macros
│   └── src/
│       ├── schema.hpp                 # ChunkNode, ObjectRecord structs
│       ├── chunk_store.cpp/.hpp       # Flat-file chunk persistence
│       ├── hnsw_index.cpp/.hpp        # HNSW ANN graph
│       ├── page_index.cpp/.hpp        # doc/page metadata map
│       ├── hybrid_ranker.cpp/.hpp     # 3-signal scorer
│       ├── kg_extractor.cpp/.hpp      # NER entity extraction
│       ├── knowledge_graph.cpp/.hpp   # Entity adjacency graph
│       ├── kg_expander.cpp/.hpp       # Multi-hop retrieval expansion
│       ├── embedder.cpp/.hpp          # ONNX embedding pipeline
│       ├── tokenizer.cpp/.hpp         # WordPiece tokenizer
│       ├── object_store.cpp/.hpp      # Relational object DB
│       ├── relation_index.cpp/.hpp    # FK relation index
│       ├── query_engine.cpp/.hpp      # Combined vector+relational query
│       ├── sync_engine.cpp/.hpp       # CRDT-based data sync
│       ├── sync_protocol.hpp          # Sync wire protocol structs
│       ├── token_budget.cpp/.hpp      # RAG context token budget
│       ├── vectordb_impl.cpp          # Main VectorDB class
│       └── vectordb_c_api.cpp         # C facade implementation
│
├── android/                           # Android SDK module
│   ├── build.gradle.kts
│   ├── CMakeLists.txt                 # Android NDK build config
│   ├── src/
│   │   ├── main/
│   │   │   ├── cpp/
│   │   │   │   └── vectordb_jni.cpp   # JNI bridge
│   │   │   ├── kotlin/
│   │   │   │   └── ai/edgevdb/
│   │   │   │       ├── EdgeVDB.kt         # Main Kotlin API
│   │   │   │       ├── VectorDBManager.kt # Lifecycle wrapper
│   │   │   │       ├── Embedder.kt        # ONNX embedding wrapper
│   │   │   │       ├── ChunkResult.kt     # Data classes
│   │   │   │       ├── ObjectRecord.kt    # Relational data classes
│   │   │   │       └── SyncConfig.kt      # Sync config data class
│   │   │   └── assets/
│   │   │       ├── model.onnx             # all-MiniLM-L6-v2 (bundled)
│   │   │       └── vocab.txt              # WordPiece vocab
│   │   └── test/
│   │       └── kotlin/ai/edgevdb/
│   │           └── EdgeVDBTest.kt
│   └── README.md
│
├── ios/                               # iOS SDK
│   ├── EdgeVDB.xcodeproj/
│   ├── Sources/
│   │   ├── EdgeVDB/
│   │   │   ├── EdgeVDB.swift          # Swift wrapper
│   │   │   ├── VectorStore.swift
│   │   │   ├── ObjectStore.swift
│   │   │   ├── Embedder.swift
│   │   │   └── SyncEngine.swift
│   │   └── EdgeVDBC/
│   │       └── bridge.h               # Bridging header → vectordb.h
│   ├── Tests/
│   │   └── EdgeVDBTests.swift
│   └── build-xcframework.sh
│
├── python/                            # Python SDK (desktop + Pi)
│   ├── edgevdb/
│   │   ├── __init__.py
│   │   ├── vectordb.py                # ctypes wrapper
│   │   ├── embedder.py                # Python embedding pipeline
│   │   ├── object_store.py            # Relational DB wrapper
│   │   └── sync.py                    # Sync wrapper
│   ├── pyproject.toml
│   ├── tests/
│   │   └── test_edgevdb.py
│   └── examples/
│       ├── pdf_rag.py
│       └── object_sync.py
│
├── models/                            # Bundled model assets
│   ├── README.md                      # Instructions to download
│   ├── download_models.sh
│   └── .gitkeep
│
├── tests/                             # C++ unit + integration tests
│   ├── CMakeLists.txt
│   ├── test_hnsw.cpp
│   ├── test_hybrid_ranker.cpp
│   ├── test_embedder.cpp
│   ├── test_object_store.cpp
│   ├── test_sync.cpp
│   ├── test_e2e_rag.cpp
│   └── benchmarks/
│       ├── bench_query_latency.cpp
│       └── bench_build_time.cpp
│
├── tools/                             # Build tooling
│   ├── convert_onnx.py                # Model quantization helper
│   └── validate_index.cpp             # Index integrity checker
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    ├── android_integration.md
    ├── ios_integration.md
    └── python_integration.md
```

---

## Phase 0 — Toolchain & Scaffolding

### Prompt 0.1 — Initialize Repository

```
Create the EdgeVDB SDK repository with the following structure exactly as specified in the
directory tree above. Initialize:

1. Root CMakeLists.txt with project name "edgevdb", version 1.0.0, C++17 standard required.
   Add subdirectories: core, tests.

2. CMakePresets.json with four presets:
   - "desktop-debug": generator Ninja, build type Debug, build dir build/desktop-debug
   - "desktop-release": generator Ninja, build type Release with -O3 -DNDEBUG
   - "android-arm64": toolchain file from NDK, ABI arm64-v8a, API level 30
   - "android-x86_64": toolchain file from NDK, ABI x86_64, API level 30

3. LICENSE file (Apache 2.0 boilerplate)

4. core/include/edgevdb/version.h defining:
   #define EDGEVDB_VERSION_MAJOR 1
   #define EDGEVDB_VERSION_MINOR 0
   #define EDGEVDB_VERSION_PATCH 0
   #define EDGEVDB_VERSION_STRING "1.0.0"

5. A root README.md with one-paragraph project description, feature table, and 
   quick-start code snippets for Android/iOS/Python.

Do NOT write any implementation code yet. Only create the scaffolding.
```

### Prompt 0.2 — Third-Party Dependencies Setup

```
Add the following third-party dependencies to the EdgeVDB project.
All dependencies must be vendored (copied into the repo) to guarantee reproducible builds
with zero internet access at build time.

1. nlohmann/json (single header):
   Download https://github.com/nlohmann/json/releases/latest single_include/nlohmann/json.hpp
   Place at: core/vendor/nlohmann/json.hpp

2. ONNX Runtime headers for C API:
   Create core/vendor/onnxruntime/ with:
   - onnxruntime_c_api.h  (the official ONNX Runtime C API header)
   Document in core/vendor/README.md that platform-specific .so/.dylib/.dll
   files must be placed here for each target platform (not committed to git).
   
3. doctest (single-header test framework):
   Place at: core/vendor/doctest/doctest.h
   This will be used for all C++ unit tests.

4. Create core/CMakeLists.txt that:
   - Defines an INTERFACE target "edgevdb_vendor" pointing to core/vendor as include path
   - Defines a STATIC library target "edgevdb_core" with all src/*.cpp files
   - Links edgevdb_core against edgevdb_vendor
   - Sets compile features cxx_std_17
   - Enables strict warnings: -Wall -Wextra -Wno-unused-parameter on GCC/Clang
   - On Android: links against log (Android logging) and android
   - On non-Android: finds and links ONNX Runtime shared library
```

---

## Phase 1 — C++ Core: Schema & ChunkStore

### Prompt 1.1 — Data Schema

```
Create core/src/schema.hpp defining all core data structures for EdgeVDB.
Use #pragma once. All structs must be standard-layout (POD-compatible).

1. ChunkNode struct:
   - uint64_t id                        // unique chunk identifier
   - char text[512]                     // null-terminated UTF-8 chunk text (fixed size for O(1) access)
   - float embedding[384]               // L2-normalised embedding vector
   - uint32_t doc_id                    // source document identifier
   - uint32_t page_number               // page within document (0-indexed)
   - uint64_t insert_timestamp          // Unix millis at insert time
   - uint8_t reserved[16]              // padding for future fields, zero-init

2. ObjectRecord struct (for relational store):
   - uint64_t id                        // unique object identifier
   - char type_name[64]                 // object type/class name
   - uint8_t data[4096]                 // msgpack-serialized property blob
   - uint64_t created_at               // Unix millis
   - uint64_t updated_at               // Unix millis
   - uint64_t sync_vector_clock        // for CRDT sync
   - bool is_deleted                   // soft delete tombstone

3. RelationEdge struct:
   - uint64_t from_id
   - uint64_t to_id
   - char relation_name[64]
   - uint64_t created_at

4. SyncEntry struct:
   - uint64_t record_id
   - char record_type[16]             // "chunk" or "object"
   - uint64_t vector_clock
   - uint8_t operation                // 0=insert, 1=update, 2=delete
   - uint8_t payload[8192]            // serialized record

5. QueryResult struct:
   - uint64_t chunk_id
   - float score
   - float cosine_score
   - float page_score
   - float keyword_score
   - uint32_t doc_id
   - uint32_t page_number
   - char text[512]

6. Define constants:
   constexpr size_t EMBEDDING_DIM = 384;
   constexpr size_t MAX_TEXT_LEN = 512;
   constexpr size_t MAX_OBJECT_DATA = 4096;
   constexpr size_t DEFAULT_TOP_K = 5;
   constexpr size_t HNSW_OVER_FETCH_FACTOR = 3;

Write a static_assert for each struct to verify it is trivially copyable.
```

### Prompt 1.2 — ChunkStore

```
Create core/src/chunk_store.hpp and core/src/chunk_store.cpp.

ChunkStore manages a flat in-memory hash map of ChunkNode records with binary file persistence.

Requirements:
1. Class ChunkStore with private members:
   - std::unordered_map<uint64_t, ChunkNode> chunks_
   - std::string file_path_
   - mutable std::shared_mutex rw_mutex_   // reader-writer lock
   - uint64_t next_id_                     // monotonic ID counter

2. Public methods:
   - ChunkStore()  // default constructor, in-memory only
   - explicit ChunkStore(const std::string& file_path)
   - bool open()   // load from file_path_ if exists, else create empty
   - bool save()   // flush all chunks to file_path_ in binary format
   - uint64_t put(const ChunkNode& chunk)  // inserts, assigns id, returns id
   - bool get(uint64_t id, ChunkNode& out) const
   - bool remove(uint64_t id)
   - size_t size() const
   - void forEach(std::function<void(const ChunkNode&)> fn) const
   - std::vector<uint64_t> getAllIds() const
   - bool clear()  // wipes all chunks and resets ID counter

3. Binary file format for chunks.bin:
   - 8-byte magic: "EVDBCHK\0"
   - 4-byte version: uint32_t = 1
   - 8-byte count: uint64_t = number of chunks
   - Followed by count * sizeof(ChunkNode) raw bytes (no padding between records)
   - 4-byte CRC32 of all chunk bytes appended at end for integrity check

4. Thread safety: all public methods must acquire appropriate lock (shared for reads, exclusive for writes)

5. Error handling: methods return bool for success/failure. Log errors via a simple
   evdb_log(level, message) macro defined in a new file core/src/log.hpp:
   #define EVDB_LOG_ERROR(msg) // on Android: __android_log_print, elsewhere: fprintf(stderr, ...)
   #define EVDB_LOG_INFO(msg)
   #define EVDB_LOG_DEBUG(msg)

Write a corresponding test file tests/test_chunk_store.cpp using doctest:
- Test: insert 1000 chunks, verify all retrievable by id
- Test: save to temp file, reload, verify identical
- Test: concurrent reads from 8 threads, verify no corruption
- Test: remove a chunk, verify get returns false
```

### Prompt 1.3 — PageIndex

```
Create core/src/page_index.hpp and core/src/page_index.cpp.

PageIndex maps chunk IDs to document/page metadata and enables page-proximity lookups.

Requirements:
1. Class PageIndex with private members:
   - std::unordered_map<uint64_t, std::pair<uint32_t,uint32_t>> chunk_to_page_
     // chunk_id → (doc_id, page_number)
   - std::unordered_map<uint32_t, std::vector<uint64_t>> doc_to_chunks_
     // doc_id → sorted list of chunk_ids in page order
   - std::string file_path_
   - mutable std::shared_mutex rw_mutex_

2. Public methods:
   - bool open()
   - bool save()
   - void insert(uint64_t chunk_id, uint32_t doc_id, uint32_t page_number)
   - bool getPage(uint64_t chunk_id, uint32_t& doc_id_out, uint32_t& page_out) const
   - std::vector<uint64_t> getChunksForDoc(uint32_t doc_id) const
   - float computePageProximity(uint64_t query_chunk_id, uint64_t neighbour_chunk_id) const
     // Returns 1.0 / (1.0 + |page_a - page_b|) if same doc, 0.0 if different doc
   - bool remove(uint64_t chunk_id)
   - void clear()

3. Binary file format for page.bin:
   - 8-byte magic: "EVDBPAG\0"
   - 4-byte version: 1
   - 8-byte entry count
   - Per entry: uint64_t chunk_id, uint32_t doc_id, uint32_t page_number
   - 4-byte CRC32 at end

Write tests/test_page_index.cpp:
- Test: insert chunks for 3 documents with varying page numbers
- Test: proximity between same-page chunks returns 1.0
- Test: proximity between page 1 and page 5 returns 1/5
- Test: proximity across documents returns 0.0
- Test: save/load round-trip preserves all entries
```

---

## Phase 2 — HNSW Vector Index

### Prompt 2.1 — HNSW Index Implementation

```
Create core/src/hnsw_index.hpp and core/src/hnsw_index.cpp (~1800 lines).
Implement a full Hierarchical Navigable Small World (HNSW) approximate nearest-neighbour
graph for 384-dimensional float32 L2-normalised vectors.

Reference: Malkov & Yashunin (2020) "Efficient and Robust Approximate Nearest Neighbor
Search Using Hierarchical Navigable Small World Graphs", IEEE Trans. PAMI.

HNSW Parameters (use these exact defaults):
  M = 16                  // max connections per node per layer
  M0 = 32                 // max connections for layer 0 (= 2*M)
  ef_construction = 200   // candidate list size during construction
  ef_search = 64          // candidate list size during search (runtime configurable)
  mL = 1.0 / ln(M)       // level generation normalizer

Data structures:
1. Node struct (internal):
   - uint64_t external_id          // maps to ChunkNode id
   - int level                     // max layer for this node
   - std::vector<std::vector<uint32_t>> neighbors  // neighbors[layer] = list of internal node indices

2. HNSWIndex class:
   - std::vector<Node> nodes_
   - std::unordered_map<uint64_t, uint32_t> id_to_index_
   - uint32_t entry_point_         // internal index of current entry point (top layer node)
   - int max_layer_               // current highest layer in graph
   - mutable std::shared_mutex rw_mutex_
   - int M_, M0_, ef_construction_, ef_search_
   - std::string file_path_

3. Core algorithms to implement:
   a. float dotProduct(const float* a, const float* b, size_t dim)
      - Use SIMD-friendly loop: process 8 floats per iteration using manual loop unrolling
      - For arm64: use ARM NEON intrinsics if __ARM_NEON is defined
      - For x86: use SSE2 intrinsics if __SSE2__ is defined
      - Fallback: scalar loop

   b. int randomLevel()
      - Returns floor(-ln(uniform01()) * mL), capped at max_layer + 1

   c. std::priority_queue searchLayer(const float* query, uint32_t entry_pt, int ef, int layer)
      - Returns ef nearest neighbours at given layer using greedy beam search
      - Uses max-heap for candidates, min-heap for results
      - Marks visited nodes with a generation counter (avoid set allocation)

   d. void selectNeighbors(candidates, M_max) → std::vector<uint32_t>
      - Simple heuristic: just return M_max closest from candidates
      - (Advanced: implement the full HNSW neighbor selection heuristic that prefers
        spread-out neighbors over clustered ones — improves recall ~2%)

   e. void insert(uint64_t external_id, const float* embedding)
      - Draw level l = randomLevel()
      - From top layer down to l+1: greedy descent, track entry point
      - From l down to 0: searchLayer(ef=ef_construction), selectNeighbors, set bidirectional links
      - Update entry_point_ if new node has higher level
      - Acquire exclusive lock only during node/edge mutation (not during search)

   f. std::vector<std::pair<uint64_t,float>> knnSearch(const float* query, int k, int ef)
      - ef defaults to ef_search_
      - Descend from top layer, then searchLayer at layer 0
      - Returns k nearest (external_id, cosine_distance) pairs, sorted ascending by distance

4. Serialization:
   Binary format for hnsw.bin:
   - 8-byte magic: "EVDBHNS\0"
   - 4-byte version: 1
   - 4-byte M, M0, ef_construction
   - 4-byte node count
   - 4-byte entry_point internal index
   - 4-byte max_layer
   - Per node: uint64_t external_id, int32_t level, then per layer: int32_t count, then count*uint32_t neighbors
   - CRC32 at end

5. Public methods:
   - bool open()
   - bool save()
   - void insert(uint64_t id, const float* embedding)
   - bool remove(uint64_t id)   // marks node as deleted (soft delete), rebuild on next open
   - std::vector<std::pair<uint64_t,float>> knnSearch(const float* query, int k) const
   - size_t size() const
   - void setEfSearch(int ef)
   - void clear()
   - bool rebuildFromChunkStore(const ChunkStore& store) // full reindex

6. Thread safety: shared_lock for knnSearch, unique_lock for insert/remove

Write tests/test_hnsw.cpp:
- Build index of 10000 random 384-dim vectors (seed=42 for reproducibility)
- Verify Recall@10 >= 0.90 (ground truth via brute force for 100 queries)
- Verify save/load gives identical knnSearch results
- Benchmark: 100 queries at 10k vectors, assert median < 50ms on debug build
- Test concurrent inserts from 4 threads: no deadlock, final size correct
```

---

## Phase 3 — Hybrid Ranker & Page Proximity

### Prompt 3.1 — Hybrid Ranker

```
Create core/src/hybrid_ranker.hpp and core/src/hybrid_ranker.cpp.

HybridRanker combines three retrieval signals to re-rank HNSW candidates.

Scoring formula:
  score(c) = α * cosine(q, c) + β * page_prox(c, best_neighbour) + γ * keyword(q, c)

Default weights: α=0.70, β=0.20, γ=0.10 (must be configurable at runtime)

1. Class HybridRanker:
   Constructor: HybridRanker(float alpha=0.70f, float beta=0.20f, float gamma=0.10f)

2. Struct RankerInput:
   - std::vector<std::pair<uint64_t, float>> hnsw_results  // (chunk_id, cosine_distance from HNSW)
   - const float* query_embedding           // 384-dim query vector
   - std::string query_text                 // raw query for keyword overlap
   - const ChunkStore* chunk_store          // to retrieve chunk text
   - const PageIndex* page_index            // for page proximity computation
   - int top_k                              // how many to return after re-ranking

3. Re-ranking algorithm:
   a. Convert HNSW cosine_distance to cosine_similarity: sim = 1.0 - distance
      (valid since vectors are L2-normalised and distances are in [0,2])
   
   b. Find best_neighbour_chunk_id = chunk_id with highest cosine similarity in candidates
   
   c. For each candidate chunk:
      - cosine_score = sim (from step a)
      - page_score = page_index->computePageProximity(chunk_id, best_neighbour_chunk_id)
        * If chunk_id == best_neighbour, page_score = 1.0
      - keyword_score = computeKeywordOverlap(query_text, chunk.text)
      - final_score = α*cosine_score + β*page_score + γ*keyword_score

   d. Sort by final_score descending, return top_k QueryResult structs

4. float computeKeywordOverlap(const std::string& query, const std::string& chunk_text):
   - Lowercase both strings
   - Tokenize on whitespace and punctuation (simple regex split or manual scan)
   - Remove stopwords (embed a small hardcoded set: "the","a","an","is","in","of","and","to","for","with","on","at","by")
   - Compute Jaccard coefficient: |intersection| / |union| of the two word bags
   - Return value in [0, 1]

5. Public method:
   std::vector<QueryResult> rerank(const RankerInput& input) const

6. Weights must be settable via:
   void setWeights(float alpha, float beta, float gamma)
   void resetWeights()  // restore defaults

7. Stopwords list: embed as a constexpr std::array<const char*, 50> covering common English stopwords.

Write tests/test_hybrid_ranker.cpp:
- Create 20 fake ChunkNodes with known text and embeddings
- Verify that a chunk on the same page as the top cosine hit is boosted vs a far-page chunk
- Verify keyword overlap returns 1.0 for identical bags, 0.0 for disjoint bags
- Verify α=1.0, β=0.0, γ=0.0 produces pure cosine ranking
```

---

## Phase 4 — Knowledge Graph Layer

### Prompt 4.1 — KG Extractor

```
Create core/src/kg_extractor.hpp and core/src/kg_extractor.cpp.

KGExtractor performs lightweight rule-based Named Entity Recognition over chunk text.
It must have zero external dependencies and run in sub-millisecond time per chunk.

Entity types to extract:
1. PROPER_NOUN: Consecutive capitalized words (e.g., "Natural Language Processing", "Apple Inc")
2. TECHNICAL_TERM: Words matching a configurable keyword set (domain-specific terms)
3. NGRAM_KEYPHRASE: Top-N TF-IDF-like bigrams/trigrams by frequency (computed at corpus level)

1. Struct Entity:
   - std::string text           // normalized entity surface form (lowercased)
   - std::string type           // "PROPER_NOUN" | "TECHNICAL_TERM" | "NGRAM"
   - uint32_t start_pos         // character offset in chunk
   - uint32_t end_pos
   - float confidence           // 0.0–1.0

2. Class KGExtractor:
   Constructor: KGExtractor()
   
   void addDomainTerms(const std::vector<std::string>& terms)
   // Add domain-specific technical vocabulary for TECHNICAL_TERM recognition

   std::vector<Entity> extract(const std::string& text) const
   // Main extraction. Steps:
   // a. Split into sentences (split on '.', '!', '?')
   // b. For each token run:
   //    - Proper noun detection: token starts with uppercase, not first in sentence
   //    - Multi-word proper noun: merge consecutive capitalized tokens up to 5 words
   //    - Technical term: exact match against domain term set (case-insensitive)
   // c. Deduplicate by normalized text, keep highest-confidence occurrence
   // d. Return up to 20 entities per chunk (cap for performance)

   std::string normalize(const std::string& text) const
   // Lowercase, trim whitespace, remove punctuation from ends

Write tests/test_kg_extractor.cpp:
- Extract from: "Dr. Alice Smith visited Stanford University. She discussed BERT models."
- Verify entities include "Alice Smith", "Stanford University", "BERT"
- Verify no extraction of common words like "She", "the"
- Performance test: 10000 extractions < 1 second total
```

### Prompt 4.2 — Knowledge Graph & Expander

```
Create core/src/knowledge_graph.hpp, knowledge_graph.cpp, kg_expander.hpp, kg_expander.cpp.

KnowledgeGraph stores entity-to-chunk and entity-to-entity co-occurrence edges.

1. Class KnowledgeGraph:
   Private:
   - std::unordered_map<std::string, std::vector<uint64_t>> entity_to_chunks_
     // entity_text → list of chunk_ids containing that entity
   - std::unordered_map<std::string, std::unordered_set<std::string>> entity_cooccurrence_
     // entity_text → set of co-occurring entity texts (from same chunk)
   - std::string file_path_
   - mutable std::shared_mutex rw_mutex_

   Public:
   - void addChunkEntities(uint64_t chunk_id, const std::vector<Entity>& entities)
     // For each entity: add chunk_id to entity_to_chunks_[entity]
     // For each pair of entities in same chunk: add co-occurrence edge (bidirectional)
   - std::vector<uint64_t> getChunksForEntity(const std::string& entity) const
   - std::unordered_set<std::string> getRelatedEntities(const std::string& entity, int hops=1) const
     // BFS up to 'hops' levels in co-occurrence graph
   - bool removeChunk(uint64_t chunk_id)
   - void clear()
   - bool save()  // binary format: "EVDBKG\0" + entries
   - bool open()

2. Class KGExpander:
   Constructor: KGExpander(const KnowledgeGraph* kg, const KGExtractor* extractor,
                            const ChunkStore* chunk_store)
   
   std::vector<uint64_t> expand(const std::vector<QueryResult>& base_results, int max_expansion=5)
   // Algorithm:
   // a. Extract entities from text of top-3 base_results chunks
   // b. For each entity, do 1-hop graph traversal to get related entities
   // c. Get all chunk_ids associated with related entities
   // d. Exclude chunk_ids already in base_results
   // e. Score expansion chunks by number of matching related entities (simple count)
   // f. Return top max_expansion unique chunk_ids sorted by score

Write tests/test_knowledge_graph.cpp:
- Insert 10 chunks with overlapping entities
- Verify 1-hop expansion finds related chunks
- Verify expansion excludes already-retrieved base chunks
- Save/load round-trip test
```

---

## Phase 5 — Embedding Pipeline (Text → Vector)

### Prompt 5.1 — WordPiece Tokenizer

```
Create core/src/tokenizer.hpp and core/src/tokenizer.cpp.

Implement a WordPiece tokenizer compatible with all-MiniLM-L6-v2 (BERT tokenizer).
This tokenizer converts raw text to token IDs for ONNX inference.

Requirements:
1. Load vocabulary from a vocab.txt file (one token per line, line index = token id)
   - [CLS] token id = 101
   - [SEP] token id = 102
   - [PAD] token id = 0
   - [UNK] token id = 100
   - Max vocab size to support: 30,000+ tokens

2. Class WordPieceTokenizer:
   explicit WordPieceTokenizer(const std::string& vocab_path)
   bool loadVocab(const std::string& vocab_path)
   
   struct TokenizerOutput {
     std::vector<int64_t> input_ids        // token ids, padded to max_length
     std::vector<int64_t> attention_mask   // 1 for real tokens, 0 for padding
     std::vector<int64_t> token_type_ids   // all zeros for single sequence
     int actual_length                     // number of real tokens (incl CLS/SEP)
   };
   
   TokenizerOutput encode(const std::string& text, int max_length=128) const
   // Steps:
   // a. Basic tokenize: lowercase, strip accents, split on whitespace+punctuation
   // b. For each basic token: WordPiece segmentation
   //    - Try to match longest prefix in vocab
   //    - If no match, output [UNK]
   //    - Continuation pieces prefixed with "##" (e.g., "##ing")
   // c. Prepend [CLS], append [SEP]
   // d. Truncate to max_length-2 tokens if necessary (truncate from right)
   // e. Pad with [PAD] to max_length
   // f. Fill attention_mask: 1 for real, 0 for pad

3. Helper: std::string normalizeText(const std::string& text)
   - Convert to lowercase
   - Unicode NFD normalization (implement basic ASCII-only version; for full Unicode, 
     use a lookup table for common accents up to U+024F)
   - Strip combining characters (accent marks)

4. The tokenizer must work identically to HuggingFace's BertTokenizerFast for ASCII text.
   This is critical: embeddings must match those from the ONNX model's expected input.

Write tests/test_tokenizer.cpp:
- Encode "Hello world" → verify [CLS, 7592, 2088, SEP] + padding
- Encode "" (empty string) → verify [CLS, SEP] + padding
- Encode a 200-word string → verify truncation at 126 tokens (max 128 - 2 for CLS/SEP)
- Verify attention_mask has 1s for real tokens and 0s for padding
- Verify token_type_ids are all zeros
```

### Prompt 5.2 — ONNX Embedding Engine

```
Create core/src/embedder.hpp and core/src/embedder.cpp.

OnnxEmbedder loads all-MiniLM-L6-v2 ONNX model and produces 384-dim L2-normalised embeddings.

Requirements:
1. Class OnnxEmbedder:
   Private:
   - OrtEnv* env_
   - OrtSession* session_
   - OrtSessionOptions* session_options_
   - OrtMemoryInfo* memory_info_
   - std::unique_ptr<WordPieceTokenizer> tokenizer_
   - bool initialized_

   Public:
   - OnnxEmbedder()
   - ~OnnxEmbedder()   // must release all ORT handles in correct order
   
   bool initialize(const std::string& model_path, const std::string& vocab_path,
                   int num_threads = 2)
   // Load ONNX model with OrtCreateSession
   // Set inter_op_num_threads and intra_op_num_threads to num_threads
   // On Android: use NNAPI execution provider if available
   //   OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, 0)
   // On all platforms: also set CPU execution provider as fallback
   
   bool embed(const std::string& text, float* output_embedding_384)
   // Steps:
   // a. tokenizer_->encode(text, 128) → input_ids, attention_mask, token_type_ids
   // b. Create 3 OrtValue inputs (shape [1, 128]):
   //    - "input_ids" (INT64)
   //    - "attention_mask" (INT64)
   //    - "token_type_ids" (INT64)
   // c. Run session: output name "last_hidden_state" (shape [1, 128, 384])
   // d. Mean pooling: for each of 384 dims, average over non-padding token positions
   //    (use attention_mask to mask padding)
   // e. L2-normalize the resulting 384-dim vector in-place
   // f. Copy into output_embedding_384
   // Return true on success, false on ORT error

   bool embedBatch(const std::vector<std::string>& texts,
                   std::vector<std::array<float,384>>& out)
   // Run batch inference: pad all to max_length in batch, single ORT call
   // Batch size limited to 32; caller should split larger batches

   bool isInitialized() const
   void shutdown()

2. Memory management:
   - Use RAII wrappers for OrtEnv, OrtSession, OrtValue
   - On Android, model path may be inside APK assets — caller must extract to app files dir first
   - Cache frequently-used OrtMemoryInfo and OrtAllocator

3. Error handling:
   - Check OrtStatus* after every ORT call
   - On error: call OrtGetErrorMessage, log via EVDB_LOG_ERROR, release status, return false

4. L2 normalization helper (also declare in embedder.hpp for reuse):
   inline void l2Normalize(float* vec, size_t dim) {
     float norm = 0.0f;
     for(size_t i=0; i<dim; i++) norm += vec[i]*vec[i];
     norm = std::sqrt(norm);
     if(norm > 1e-8f) { for(size_t i=0; i<dim; i++) vec[i] /= norm; }
   }

Write tests/test_embedder.cpp:
- Integration test (only runs if model.onnx is present at TEST_MODEL_PATH env var)
- Verify embed("hello world") returns a 384-dim vector with L2-norm ≈ 1.0
- Verify embed("cat") and embed("dog") have higher cosine similarity than embed("cat") and embed("airplane")
- Verify embedBatch produces same results as embed called individually
- If model not present, skip with doctest [.][integration] tag
```

---

## Phase 6 — Relational Object Store

> **Why this is needed:** Pure vector databases like FAISS and sqlite-vec store only embeddings.
> Real applications need to store structured metadata alongside vectors — user profiles, document
> metadata, chat history, application state — and query them with typed conditions. ObjectBox
> provides this (it is fundamentally a NoSQL object database with vector search bolted on).
> EdgeVDB must match this capability.

### Prompt 6.1 — Object Store Engine

```
Create core/src/object_store.hpp and core/src/object_store.cpp.

ObjectStore is a lightweight schema-less object database storing typed records alongside vectors.
It is separate from but queryable together with the vector store.

Design: Records are stored as MessagePack-serialized blobs keyed by (type_name, id).
Use nlohmann/json as the in-memory representation; serialize to compact binary before writing.

1. Class ObjectStore:
   Private:
   - std::unordered_map<uint64_t, ObjectRecord> records_     // id → record
   - std::unordered_map<std::string, std::vector<uint64_t>> type_index_  // type_name → ids
   - std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string,std::vector<uint64_t>>>> property_index_
     // type_name → property_name → property_value_str → [ids]
   - std::string file_path_
   - mutable std::shared_mutex rw_mutex_
   - uint64_t next_id_

   Public:
   - bool open()
   - bool save()
   
   uint64_t put(const std::string& type_name, const nlohmann::json& properties)
   // Creates or upserts a record. Assigns new id if properties has no "id".
   // If properties["id"] exists, treat as update.
   // Serializes properties to JSON string stored in ObjectRecord.data
   // Updates type_index_ and property_index_ for all string/number properties
   // Returns assigned id

   bool get(uint64_t id, nlohmann::json& out) const
   // Deserializes record, returns false if not found or soft-deleted

   bool remove(uint64_t id)
   // Sets is_deleted = true (tombstone for sync). Does NOT erase from map.

   std::vector<nlohmann::json> query(const std::string& type_name,
                                      const std::string& property,
                                      const std::string& value,
                                      int limit = 100) const
   // Returns all records of type_name where property == value
   // Uses property_index_ for O(1) lookup

   std::vector<nlohmann::json> getAll(const std::string& type_name,
                                       int offset = 0, int limit = 100) const

   std::vector<uint64_t> getIdsForType(const std::string& type_name) const

   size_t count(const std::string& type_name) const
   size_t totalCount() const

2. Binary file format for objects.bin:
   - 8-byte magic "EVDBOBJ\0"
   - 4-byte version 1
   - 8-byte record count
   - Per record: sizeof(ObjectRecord) raw bytes
   - 4-byte CRC32

3. The ObjectRecord.data field stores the JSON as a UTF-8 string (null-terminated, max 4096 bytes).
   Caller is responsible for not exceeding this limit.

4. Indexing: on open(), rebuild type_index_ and property_index_ from all non-deleted records.

Write tests/test_object_store.cpp:
- Insert 500 objects of type "Document" with properties {title, author, page_count}
- Insert 300 objects of type "User" with properties {name, email}
- query("Document", "author", "Alice") returns correct subset
- getAll("User", offset=100, limit=50) returns correct slice
- remove() then get() returns false
- save/load round-trip: all records preserved
```

### Prompt 6.2 — Relation Index

```
Create core/src/relation_index.hpp and core/src/relation_index.cpp.

RelationIndex stores typed edges between ObjectStore records (foreign key relations).

1. Class RelationIndex:
   - std::unordered_map<std::string, std::vector<std::pair<uint64_t,uint64_t>>> relation_edges_
     // relation_name → [(from_id, to_id)]
   - std::unordered_map<std::string, std::unordered_map<uint64_t, std::vector<uint64_t>>> from_index_
     // relation_name → from_id → [to_ids]
   - std::unordered_map<std::string, std::unordered_map<uint64_t, std::vector<uint64_t>>> to_index_
     // relation_name → to_id → [from_ids]

   Public:
   - void addRelation(const std::string& name, uint64_t from_id, uint64_t to_id)
   - std::vector<uint64_t> getTargets(const std::string& name, uint64_t from_id) const
   - std::vector<uint64_t> getSources(const std::string& name, uint64_t to_id) const
   - bool removeRelation(const std::string& name, uint64_t from_id, uint64_t to_id)
   - void removeAllRelationsForRecord(uint64_t record_id)  // called on object delete
   - bool save(const std::string& path) const
   - bool open(const std::string& path)
   - void clear()

Example usage this enables:
  // Link a "Chunk" to its source "Document"
  relation_index.addRelation("sourceDocument", chunk_id, doc_object_id);
  // Get all chunks from a document
  auto chunk_ids = relation_index.getTargets("sourceDocument", doc_object_id);
  // reversed: get the document object for a chunk
  auto doc_ids = relation_index.getSources("sourceDocument", chunk_id);

Write tests for all methods.
```

### Prompt 6.3 — Combined Query Engine

```
Create core/src/query_engine.hpp and core/src/query_engine.cpp.

QueryEngine provides unified queries spanning both vector search and relational object store.

1. Struct CombinedQuery:
   std::string text_query              // used for vector search and keyword overlap
   std::optional<std::string> filter_type_name    // restrict to chunks linked to objects of this type
   std::optional<std::string> filter_property     // filter property name
   std::optional<std::string> filter_value        // filter property value
   int top_k = 5
   bool use_kg_expansion = false
   int token_budget = 3200             // max tokens in assembled context
   float token_estimate_ratio = 3.5f  // chars per token estimate

2. Struct CombinedResult:
   std::vector<QueryResult> chunks           // ranked retrieved chunks
   std::string assembled_context            // ready-to-inject RAG context string
   bool context_truncated                   // true if token budget was hit
   int total_tokens_estimated

3. Class QueryEngine:
   Constructor:
   QueryEngine(VectorDB* vdb, ObjectStore* obj_store, RelationIndex* rel_index)
   
   CombinedResult query(const CombinedQuery& q) const
   // Steps:
   // a. vdb->embed(q.text_query) → query_vector
   // b. vdb->knnSearch(query_vector, q.top_k * OVER_FETCH) → raw_candidates
   // c. If filter_type_name set:
   //    - Get all ids linked via any relation to objects matching (filter_type_name, filter_property, filter_value)
   //    - Intersect raw_candidates with these ids (keep only matching chunks)
   // d. Hybrid re-rank
   // e. If use_kg_expansion: expand via KGExpander, add expansion chunks after ranked results
   // f. Assemble context string: concatenate chunk.text with [Page N] prefix, honoring token_budget
   //    When budget exhausted, append "[Context truncated for length]" marker

   std::string assembleContext(const std::vector<QueryResult>& results,
                                int token_budget, float token_ratio,
                                bool& truncated_out) const
   // Builds the RAG context string

Write tests/test_query_engine.cpp testing the combined pipeline.
```

---

## Phase 7 — Data Sync Engine

> **Why this is needed & why it was missing from EdgeVDB paper:**
> The EdgeVDB paper is scoped to single-device RAG. Data sync is architecturally
> separate and substantially complex. It was missing because:
> (1) It requires a wire protocol and conflict-resolution policy beyond single-device scope,
> (2) ObjectBox spent years building their sync infrastructure.
> However, for a production SDK competing with ObjectBox, sync is table-stakes.
> We implement it using Vector Clocks + Last-Write-Wins CRDTs — the simplest correct
> approach for eventual consistency without requiring a central server.

### Prompt 7.1 — Sync Engine (CRDT-based)

```
Create core/src/sync_engine.hpp and core/src/sync_engine.cpp.

SyncEngine enables bi-directional data sync between EdgeVDB instances across devices.
It uses Vector Clock + Last-Write-Wins (LWW) semantics per record.

Design principles:
- No central server required (peer-to-peer capable)
- Eventual consistency with LWW conflict resolution
- Sync vector_clock per ObjectRecord tracks causality
- Network transport is NOT implemented here — SyncEngine produces/consumes serialized
  change deltas. The transport layer (TCP socket, WebSocket, BLE, file export) is
  injected by the caller.

1. Struct DeviceClock:
   - std::string device_id     // UUID assigned on first launch (stored in config)
   - uint64_t logical_clock    // monotonically increasing per-device event counter
   
   uint64_t tick()             // increment and return logical_clock
   bool happensAfter(uint64_t a, uint64_t b) const  // a > b for LWW

2. Struct SyncDelta:
   - std::string source_device_id
   - uint64_t from_clock        // send records with vector_clock > from_clock
   - uint64_t to_clock          // highest clock in this delta
   - std::vector<ObjectRecord> records   // changed records since from_clock
   - std::vector<RelationEdge> edges     // changed edges since from_clock
   
   std::string serialize() const    // JSON via nlohmann/json
   static SyncDelta deserialize(const std::string& json)

3. Class SyncEngine:
   Constructor: SyncEngine(const std::string& device_id,
                            ObjectStore* obj_store,
                            RelationIndex* rel_index,
                            ChunkStore* chunk_store)
   
   SyncDelta exportDelta(uint64_t since_clock) const
   // Collect all ObjectRecords with vector_clock > since_clock
   // Collect all RelationEdges modified after since_clock
   // Collect all ChunkNodes with insert_timestamp > since_clock (treated as clock)
   // Return as SyncDelta
   
   struct MergeResult {
     int records_applied;
     int records_skipped_older;   // remote record was older than local, skipped
     int conflicts_lww_resolved;  // both modified; took higher clock
   };
   
   MergeResult applyDelta(const SyncDelta& delta)
   // For each record in delta:
   //   - If not in local store: insert
   //   - If in local store: compare vector_clocks
   //     * remote.vector_clock > local.vector_clock → apply remote (LWW)
   //     * remote.vector_clock <= local.vector_clock → skip (local is newer)
   //     * equal clocks, different content → tie-break by device_id lexicographic order
   // For each edge: apply if not already present
   // For each chunk: apply if not already present (chunks are immutable once inserted)
   
   uint64_t getCurrentClock() const
   void onRecordMutated(uint64_t record_id)  // call after every put/remove to tick clock

4. Sync configuration:
   Struct SyncConfig:
   - bool enabled = false
   - std::string device_id           // auto-generated UUID if empty
   - std::string sync_endpoint       // WebSocket URL or "" for local file sync
   - int sync_interval_seconds = 60
   - bool sync_chunks = true         // whether to sync vector chunks
   - bool sync_objects = true        // whether to sync object records

5. File-based sync (for Pi/desktop and offline mobile):
   bool exportToFile(const std::string& path, uint64_t since_clock) const
   bool importFromFile(const std::string& path, MergeResult& result)
   // Serializes SyncDelta to/from JSON file for manual sync

Write tests/test_sync.cpp:
- Create two SyncEngine instances with different device_ids
- Insert 100 records in device A, export delta
- Apply delta to device B — verify all 100 records present
- Modify same record in both A and B with different values
- Apply each other's delta — verify LWW resolution picks correct winner
- Test idempotency: apply same delta twice → same result
```

---

## Phase 8 — Public C API Facade

### Prompt 8.1 — C API Header

```
Create core/include/edgevdb/vectordb.h — the stable public C API for EdgeVDB.

This header is the ONLY thing external consumers include. It must:
- Be valid C99 and C++17
- Use opaque handle types (no struct internals exposed)
- Have a stable ABI (no C++ classes, no templates, no exceptions)
- Be exhaustively documented with Doxygen comments

Define the following:

/* ── Handle types ──────────────────────────────────────── */
typedef struct EvdbHandle_       EvdbHandle;       /* main database handle */
typedef struct EvdbQueryHandle_  EvdbQueryHandle;  /* query result handle */
typedef struct EvdbEmbedder_     EvdbEmbedder;     /* embedder handle */
typedef struct EvdbSyncEngine_   EvdbSyncEngine;   /* sync engine handle */

/* ── Error codes ───────────────────────────────────────── */
typedef enum {
  EVDB_OK              = 0,
  EVDB_ERR_NULL_HANDLE = 1,
  EVDB_ERR_IO          = 2,
  EVDB_ERR_OOM         = 3,
  EVDB_ERR_NOT_FOUND   = 4,
  EVDB_ERR_INVALID_ARG = 5,
  EVDB_ERR_ONNX        = 6,
  EVDB_ERR_SYNC        = 7,
} EvdbError;

/* ── Config struct ─────────────────────────────────────── */
typedef struct {
  const char* storage_dir;         /* directory for .bin files */
  int hnsw_M;                      /* default 16 */
  int hnsw_ef_construction;        /* default 200 */
  int hnsw_ef_search;              /* default 64 */
  float ranker_alpha;              /* cosine weight, default 0.70 */
  float ranker_beta;               /* page proximity weight, default 0.20 */
  float ranker_gamma;              /* keyword weight, default 0.10 */
  int token_budget;                /* default 3200 */
  int embedding_threads;           /* ONNX thread count, default 2 */
  int enable_knowledge_graph;      /* 0 or 1, default 1 */
  int enable_sync;                 /* 0 or 1, default 0 */
  const char* device_id;           /* for sync, NULL = auto-generate */
} EvdbConfig;

/* ── Lifecycle ─────────────────────────────────────────── */
EvdbHandle* evdb_open(const EvdbConfig* config);
void        evdb_close(EvdbHandle* h);
EvdbError   evdb_save(EvdbHandle* h);
void        evdb_default_config(EvdbConfig* out);  /* fills safe defaults */

/* ── Embedding pipeline ────────────────────────────────── */
EvdbEmbedder* evdb_embedder_create(const char* model_path, const char* vocab_path, int threads);
void          evdb_embedder_destroy(EvdbEmbedder* e);
EvdbError     evdb_embed_text(EvdbEmbedder* e, const char* text, float* out_384);

/* ── Vector store — insert ─────────────────────────────── */
EvdbError evdb_insert_text(EvdbHandle* h, EvdbEmbedder* e,
                            const char* text, uint32_t doc_id, uint32_t page_number,
                            uint64_t* out_chunk_id);
/* Insert pre-computed embedding: */
EvdbError evdb_insert_chunk(EvdbHandle* h,
                             const char* text, const float* embedding_384,
                             uint32_t doc_id, uint32_t page_number,
                             uint64_t* out_chunk_id);
EvdbError evdb_remove_chunk(EvdbHandle* h, uint64_t chunk_id);

/* ── Vector store — query ──────────────────────────────── */
EvdbQueryHandle* evdb_query_text(EvdbHandle* h, EvdbEmbedder* e,
                                  const char* query_text, int top_k,
                                  int use_kg_expansion);
EvdbQueryHandle* evdb_query_vector(EvdbHandle* h,
                                    const float* query_embedding_384,
                                    const char* query_text_for_keyword,
                                    int top_k);

int         evdb_result_count(EvdbQueryHandle* q);
const char* evdb_result_text(EvdbQueryHandle* q, int index);
float       evdb_result_score(EvdbQueryHandle* q, int index);
uint64_t    evdb_result_chunk_id(EvdbQueryHandle* q, int index);
uint32_t    evdb_result_page(EvdbQueryHandle* q, int index);
const char* evdb_result_context_string(EvdbQueryHandle* q);  /* assembled RAG context */
void        evdb_query_free(EvdbQueryHandle* q);

/* ── Object store ──────────────────────────────────────── */
EvdbError evdb_object_put(EvdbHandle* h,
                           const char* type_name,
                           const char* json_properties,
                           uint64_t* out_id);
EvdbError evdb_object_get(EvdbHandle* h, uint64_t id, char* out_json, int out_json_size);
EvdbError evdb_object_remove(EvdbHandle* h, uint64_t id);
EvdbError evdb_object_query(EvdbHandle* h,
                             const char* type_name,
                             const char* filter_property,
                             const char* filter_value,
                             char* out_json_array,
                             int out_size,
                             int limit);

/* ── Relations ─────────────────────────────────────────── */
EvdbError evdb_relation_add(EvdbHandle* h,
                             const char* relation_name,
                             uint64_t from_id, uint64_t to_id);
EvdbError evdb_relation_get_targets(EvdbHandle* h,
                                     const char* relation_name,
                                     uint64_t from_id,
                                     uint64_t* out_ids, int* inout_count);

/* ── Sync ──────────────────────────────────────────────── */
EvdbSyncEngine* evdb_sync_create(EvdbHandle* h, const char* device_id);
void            evdb_sync_destroy(EvdbSyncEngine* s);
EvdbError       evdb_sync_export_delta(EvdbSyncEngine* s,
                                        uint64_t since_clock,
                                        char* out_json, int out_size);
EvdbError       evdb_sync_apply_delta(EvdbSyncEngine* s,
                                       const char* delta_json,
                                       int* out_applied, int* out_skipped);
EvdbError       evdb_sync_export_to_file(EvdbSyncEngine* s,
                                          const char* path, uint64_t since_clock);
EvdbError       evdb_sync_import_from_file(EvdbSyncEngine* s, const char* path);
uint64_t        evdb_sync_current_clock(EvdbSyncEngine* s);

/* ── Utilities ─────────────────────────────────────────── */
const char* evdb_version_string(void);
const char* evdb_error_string(EvdbError err);
void        evdb_set_log_level(int level);   /* 0=off, 1=error, 2=info, 3=debug */

#ifdef __cplusplus
}
#endif
#endif /* VECTORDB_H */
```

### Prompt 8.2 — C API Implementation

```
Create core/src/vectordb_c_api.cpp implementing all functions declared in vectordb.h.

Each EvdbHandle* is a heap-allocated struct containing:
  struct EvdbHandle_ {
    std::unique_ptr<ChunkStore>     chunk_store;
    std::unique_ptr<HNSWIndex>      hnsw_index;
    std::unique_ptr<PageIndex>      page_index;
    std::unique_ptr<HybridRanker>   ranker;
    std::unique_ptr<KGExtractor>    kg_extractor;
    std::unique_ptr<KnowledgeGraph> kg;
    std::unique_ptr<KGExpander>     kg_expander;
    std::unique_ptr<ObjectStore>    obj_store;
    std::unique_ptr<RelationIndex>  rel_index;
    std::unique_ptr<QueryEngine>    query_engine;
    EvdbConfig config;
    std::string storage_dir;
  };

Each EvdbEmbedder* is a heap-allocated OnnxEmbedder.
Each EvdbQueryHandle* holds a vector<QueryResult> and the assembled_context string.
Each EvdbSyncEngine* holds a SyncEngine.

Implementation rules:
1. Every function must null-check its handle parameter first; return EVDB_ERR_NULL_HANDLE
2. Wrap every call in try/catch(std::exception&); log and return EVDB_ERR_IO
3. String output parameters (char* out_json): always null-terminate, never overflow buffer
4. evdb_open():
   - Allocate EvdbHandle_
   - Construct each component with storage_dir prepended to file names
   - Call .open() on ChunkStore, HNSWIndex, PageIndex, KnowledgeGraph, ObjectStore, RelationIndex
   - Return handle or nullptr on failure
5. evdb_close(): call evdb_save() then delete handle
6. evdb_insert_text(): calls embedder to get vector, then evdb_insert_chunk()
7. evdb_query_text(): calls embedder, then knnSearch with OVER_FETCH, then HybridRanker, 
   then optionally KGExpander, then assembleContext, stores in EvdbQueryHandle

Write tests/test_c_api.cpp:
- Full end-to-end test using only the C API (no internal C++ headers)
- Insert 50 text chunks via evdb_insert_text
- Query and verify top result is correct
- Put/get/query object store via C API
- Add relation and retrieve targets
```

---

## Phase 9 — Android SDK (JNI + Kotlin)

### Prompt 9.1 — JNI Bridge

```
Create android/src/main/cpp/vectordb_jni.cpp.

This file bridges the C API to Java/Kotlin via JNI. Every JNI function maps to one
or more evdb_* C API calls.

JNI function naming convention: Java_ai_edgevdb_EdgeVDB_<methodName>

Implement JNI functions for:
1. nativeOpen(String storageDir, int hnswM, int efConstruction, int efSearch,
              float alpha, float beta, float gamma, int tokenBudget) → Long (handle pointer)
2. nativeClose(long handle)
3. nativeSave(long handle) → Int (EvdbError)
4. nativeEmbedderCreate(String modelPath, String vocabPath, int threads) → Long
5. nativeEmbedderDestroy(long embedderHandle)
6. nativeInsertText(long handle, long embedderHandle, String text,
                    int docId, int pageNumber) → Long (chunk_id, -1 on error)
7. nativeInsertChunk(long handle, FloatArray embedding, String text,
                     int docId, int pageNumber) → Long
8. nativeRemoveChunk(long handle, long chunkId) → Int
9. nativeQueryText(long handle, long embedderHandle, String queryText,
                   int topK, boolean useKgExpansion) → LongArray (query handle)
10. nativeResultCount(long queryHandle) → Int
11. nativeResultText(long queryHandle, int index) → String
12. nativeResultScore(long queryHandle, int index) → Float
13. nativeResultPage(long queryHandle, int index) → Int
14. nativeResultContextString(long queryHandle) → String
15. nativeQueryFree(long queryHandle)
16. nativeObjectPut(long handle, String typeName, String jsonProperties) → Long
17. nativeObjectGet(long handle, long id) → String (JSON)
18. nativeObjectRemove(long handle, long id) → Int
19. nativeObjectQuery(long handle, String typeName, String property,
                      String value, int limit) → String (JSON array)
20. nativeRelationAdd(long handle, String name, long fromId, long toId) → Int
21. nativeSyncExportDelta(long handle, String deviceId, long sinceClock) → String (JSON)
22. nativeSyncApplyDelta(long handle, String deviceId, String deltaJson) → IntArray [applied, skipped]
23. nativeVersion() → String

JNI best practices to follow:
- Cache jclass and jmethodID references in JNI_OnLoad
- Use GetStringUTFChars/ReleaseStringUTFChars with isCopy check
- Use NewStringUTF for return strings, check for OOM (returns null)
- All exceptions must be caught: wrap every call in try/catch and throw back via
  env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what())
- Log to Android logcat via __android_log_print with tag "EdgeVDB"
```

### Prompt 9.2 — Kotlin SDK

```
Create the Kotlin SDK in android/src/main/kotlin/ai/edgevdb/.

1. EdgeVDB.kt — Main entry point:
   class EdgeVDB private constructor(private val handle: Long) {
     companion object {
       fun open(context: Context, config: EdgeVDBConfig = EdgeVDBConfig()): EdgeVDB
       // Resolves storageDir to context.filesDir/edgevdb/
       // Calls nativeOpen with config params
       // Returns EdgeVDB instance
       
       init {
         System.loadLibrary("edgevdb")  // load the .so
       }
     }
     
     fun close()
     fun save(): Boolean
     
     // Vector store
     fun insertText(embedder: Embedder, text: String, docId: Int, pageNumber: Int): Long
     fun insertChunk(embedding: FloatArray, text: String, docId: Int, pageNumber: Int): Long
     fun removeChunk(chunkId: Long): Boolean
     
     // Query
     fun queryText(embedder: Embedder, queryText: String,
                   topK: Int = 5, useKgExpansion: Boolean = false): QueryResults
     
     // Object store
     fun putObject(typeName: String, properties: Map<String, Any>): Long
     fun getObject(id: Long): Map<String, Any>?
     fun removeObject(id: Long): Boolean
     fun queryObjects(typeName: String, property: String, value: String, limit: Int = 100): List<Map<String, Any>>
     
     // Relations
     fun addRelation(name: String, fromId: Long, toId: Long): Boolean
     fun getRelationTargets(name: String, fromId: Long): List<Long>
     
     // Sync
     fun exportDelta(sinceClock: Long): String
     fun applyDelta(deltaJson: String): SyncResult
   }

2. EdgeVDBConfig.kt:
   data class EdgeVDBConfig(
     val hnswM: Int = 16,
     val efConstruction: Int = 200,
     val efSearch: Int = 64,
     val rankerAlpha: Float = 0.70f,
     val rankerBeta: Float = 0.20f,
     val rankerGamma: Float = 0.10f,
     val tokenBudget: Int = 3200,
     val embeddingThreads: Int = 2,
     val enableKnowledgeGraph: Boolean = true,
     val enableSync: Boolean = false,
     val deviceId: String = UUID.randomUUID().toString()
   )

3. Embedder.kt:
   class Embedder private constructor(private val handle: Long) {
     companion object {
       // Try loading from assets first; if not found, require explicit path
       fun fromAssets(context: Context, 
                      modelAsset: String = "model.onnx",
                      vocabAsset: String = "vocab.txt",
                      threads: Int = 2): Embedder
       // Copies assets to filesDir if not already there, then calls nativeEmbedderCreate
       
       fun fromFiles(modelPath: String, vocabPath: String, threads: Int = 2): Embedder
     }
     fun embed(text: String): FloatArray   // returns 384-dim float array
     fun destroy()
   }

4. QueryResults.kt:
   data class ChunkResult(
     val chunkId: Long,
     val text: String,
     val score: Float,
     val pageNumber: Int,
     val docId: Int
   )
   
   class QueryResults internal constructor(private val handle: Long) : AutoCloseable {
     val count: Int get() = nativeResultCount(handle)
     fun get(index: Int): ChunkResult
     val contextString: String get() = nativeResultContextString(handle)
     fun toList(): List<ChunkResult>
     override fun close() = nativeQueryFree(handle)
   }

5. SyncResult.kt:
   data class SyncResult(val applied: Int, val skipped: Int)

6. VectorDBManager.kt — Lifecycle-aware wrapper for Android ViewModel use:
   class VectorDBManager(private val context: Context) {
     private var db: EdgeVDB? = null
     private var embedder: Embedder? = null
     
     suspend fun initialize(config: EdgeVDBConfig = EdgeVDBConfig()): Boolean
     // Runs on Dispatchers.IO
     // Opens EdgeVDB and Embedder
     
     suspend fun insertDocument(pdfText: List<Pair<String, Int>>): Int
     // pdfText: List of (chunk_text, page_number)
     // Auto-assigns doc_id
     // Returns count of inserted chunks
     
     suspend fun query(text: String, topK: Int = 5): QueryResults
     
     fun isInitialized(): Boolean
     fun close()
   }

7. android/CMakeLists.txt:
   - Define a SHARED library target "edgevdb"
   - Include all core/src/*.cpp files
   - Include vectordb_jni.cpp
   - Set include paths for core/include and core/vendor
   - Link against: log, android
   - Set -O2 -fvisibility=hidden -DANDROID
   - For arm64: add -mfpu=neon (enables NEON SIMD in dot product)

8. android/build.gradle.kts:
   - Library module (not application)
   - minSdk = 26, compileSdk = 35
   - NDK ABI filters: ["arm64-v8a", "x86_64"]
   - CMake version 3.22.1
   - Dependencies: onnxruntime-android (1.20.0), kotlinx-coroutines-android
   - Publishing configuration for Maven local + optional Maven Central
```

---

## Phase 10 — iOS SDK (C API + Swift)

### Prompt 10.1 — iOS Swift Wrapper

```
Create the iOS SDK in ios/Sources/EdgeVDB/.

1. EdgeVDB.swift — Main Swift class:
   public final class EdgeVDB {
     private var handle: OpaquePointer?
     
     public init(storageDir: URL, config: EdgeVDBConfig = EdgeVDBConfig()) throws
     public func close()
     public func save() throws
     
     // Vector store
     public func insertText(using embedder: Embedder, text: String,
                            docId: UInt32, pageNumber: UInt32) throws -> UInt64
     public func insertChunk(embedding: [Float], text: String,
                             docId: UInt32, pageNumber: UInt32) throws -> UInt64
     public func removeChunk(_ chunkId: UInt64) throws
     
     // Query
     public func queryText(using embedder: Embedder, query: String,
                           topK: Int = 5, useKgExpansion: Bool = false) throws -> QueryResults
     
     // Object store
     public func putObject(type: String, properties: [String: Any]) throws -> UInt64
     public func getObject(_ id: UInt64) throws -> [String: Any]?
     public func removeObject(_ id: UInt64) throws
     public func queryObjects(type: String, property: String,
                              value: String, limit: Int = 100) throws -> [[String: Any]]
     
     // Relations
     public func addRelation(_ name: String, from fromId: UInt64, to toId: UInt64) throws
     public func relationTargets(_ name: String, from fromId: UInt64) throws -> [UInt64]
     
     // Sync
     public func exportDelta(since clock: UInt64) throws -> String
     public func applyDelta(_ json: String) throws -> SyncResult
   }

2. Embedder.swift:
   public final class Embedder {
     private var handle: OpaquePointer?
     
     public init(modelURL: URL, vocabURL: URL, threads: Int = 2) throws
     // For on-device resources bundled in app bundle:
     public convenience init(modelResource: String, vocabResource: String,
                              bundle: Bundle = .main) throws
     public func embed(_ text: String) throws -> [Float]  // 384-dim
     public func destroy()
   }

3. QueryResults.swift:
   public struct ChunkResult {
     public let chunkId: UInt64
     public let text: String
     public let score: Float
     public let pageNumber: UInt32
     public let docId: UInt32
   }
   
   public final class QueryResults {
     private var handle: OpaquePointer?
     public var count: Int { get }
     public subscript(index: Int) -> ChunkResult { get }
     public var contextString: String { get }
     public func toArray() -> [ChunkResult]
     deinit { evdb_query_free(handle) }
   }

4. EdgeVDBConfig.swift:
   public struct EdgeVDBConfig {
     public var hnswM: Int32 = 16
     public var efConstruction: Int32 = 200
     public var efSearch: Int32 = 64
     public var rankerAlpha: Float = 0.70
     public var rankerBeta: Float = 0.20
     public var rankerGamma: Float = 0.10
     public var tokenBudget: Int32 = 3200
     public var embeddingThreads: Int32 = 2
     public var enableKnowledgeGraph: Bool = true
     public var enableSync: Bool = false
     public var deviceId: String = UUID().uuidString
   }

5. Error handling: define EdgeVDBError: Swift.Error with cases matching EvdbError codes

6. ios/Sources/EdgeVDBC/bridge.h:
   #include "edgevdb/vectordb.h"
   
7. ios/Package.swift (Swift Package Manager):
   .package with targets: EdgeVDB (Swift), EdgeVDBC (C)
   EdgeVDB depends on EdgeVDBC
   EdgeVDBC uses cSettings publicHeadersPath pointing to core/include

8. build-xcframework.sh:
   Build for: arm64-apple-ios, arm64-apple-ios-simulator, x86_64-apple-ios-simulator
   Merge into EdgeVDB.xcframework via xcodebuild -create-xcframework

Write ios/Tests/EdgeVDBTests.swift covering all public APIs.
```

---

## Phase 11 — Desktop/Python SDK

### Prompt 11.1 — Python SDK

```
Create the Python SDK in python/edgevdb/.

1. python/edgevdb/vectordb.py — ctypes wrapper:
   import ctypes, json, os, platform
   from pathlib import Path
   
   def _load_lib():
     # Load platform-specific .so/.dylib/.dll
     # Search order: env var EDGEVDB_LIB_PATH, package dir, system path
     ...
   
   _lib = _load_lib()
   
   # Define ctypes function signatures for every evdb_* function
   # Example:
   _lib.evdb_open.restype = ctypes.c_void_p
   _lib.evdb_open.argtypes = [ctypes.POINTER(EvdbConfig)]
   
   class EdgeVDB:
     def __init__(self, storage_dir: str, **config_kwargs):
       ...
     def __enter__(self): return self
     def __exit__(self, *args): self.close()
     
     def close(self)
     def save(self) -> bool
     
     def insert_text(self, embedder: 'Embedder', text: str,
                     doc_id: int = 0, page_number: int = 0) -> int
     def insert_chunk(self, embedding: list[float], text: str,
                      doc_id: int = 0, page_number: int = 0) -> int
     def query_text(self, embedder: 'Embedder', query: str,
                    top_k: int = 5, use_kg: bool = False) -> 'QueryResults'
     
     def put_object(self, type_name: str, properties: dict) -> int
     def get_object(self, object_id: int) -> dict | None
     def query_objects(self, type_name: str, property: str, value: str, limit: int = 100) -> list[dict]
     def remove_object(self, object_id: int) -> bool
     
     def add_relation(self, name: str, from_id: int, to_id: int) -> bool
     def relation_targets(self, name: str, from_id: int) -> list[int]
     
     def export_delta(self, since_clock: int = 0) -> str
     def apply_delta(self, delta_json: str) -> tuple[int, int]  # (applied, skipped)
   
   class Embedder:
     def __init__(self, model_path: str, vocab_path: str, threads: int = 2):
     def embed(self, text: str) -> list[float]  # 384-dim
     def destroy(self)
   
   class QueryResults:
     @property
     def count(self) -> int
     def __getitem__(self, index: int) -> dict  # {chunk_id, text, score, page_number, doc_id}
     def __len__(self) -> int
     def __iter__(self)
     @property
     def context_string(self) -> str
     def to_list(self) -> list[dict]
     def free(self)

2. python/edgevdb/__init__.py:
   Export EdgeVDB, Embedder, QueryResults, EdgeVDBConfig at package level

3. python/pyproject.toml:
   [project]
   name = "edgevdb"
   version = "1.0.0"
   dependencies = []   # zero Python deps; only standard library + ctypes
   
   [project.optional-dependencies]
   dev = ["pytest", "numpy"]

4. python/examples/pdf_rag.py:
   Complete working example:
   - Read a text file in chunks of 400 words
   - Embed each chunk and insert into EdgeVDB
   - Interactive query loop: read query from stdin, print top-5 results + context string
   
5. python/examples/object_sync.py:
   Complete example demonstrating:
   - Insert 100 objects across two EdgeVDB instances in separate directories
   - Export delta from instance A
   - Apply to instance B
   - Verify merge

6. python/tests/test_edgevdb.py:
   pytest test suite covering all Python API functions
   Mark integration tests with @pytest.mark.skipif(not model_available, ...)
```

---

## Phase 12 — CMake Unified Build System

### Prompt 12.1 — CMake Build Files

```
Create the complete CMake build system.

1. Root CMakeLists.txt:
   cmake_minimum_required(VERSION 3.22)
   project(edgevdb VERSION 1.0.0 LANGUAGES C CXX)
   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)
   
   option(EDGEVDB_BUILD_TESTS "Build tests" ON)
   option(EDGEVDB_BUILD_PYTHON "Build Python extension" OFF)
   option(EDGEVDB_ENABLE_NEON "Enable ARM NEON SIMD" ON)
   option(EDGEVDB_ENABLE_SSE2 "Enable x86 SSE2 SIMD" ON)
   option(EDGEVDB_ENABLE_SYNC "Build sync engine" ON)
   
   add_subdirectory(core)
   if(EDGEVDB_BUILD_TESTS AND NOT ANDROID)
     add_subdirectory(tests)
   endif()

2. core/CMakeLists.txt:
   Defines:
   - edgevdb_core STATIC library
   - edgevdb_shared SHARED library (same sources, different target name)
   - Install rules: headers to include/edgevdb, libs to lib/
   - Platform compile definitions:
     if(ANDROID): add -DEDGEVDB_ANDROID, link log/android
     if(APPLE and IOS): add -DEDGEVDB_IOS
     if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64"): add -DEDGEVDB_ARM64
     if(EDGEVDB_ENABLE_NEON and ARM64): add -mfpu=neon (GCC) or nothing (Clang uses auto)
     if(EDGEVDB_ENABLE_SSE2 and x86): add -msse2

3. tests/CMakeLists.txt:
   find_package(Threads REQUIRED)
   # Add test executables for each test_*.cpp
   # Each links against edgevdb_core and doctest
   # Register with CTest: add_test(NAME test_hnsw COMMAND test_hnsw)
   
   # Benchmark executables — only built in Release
   if(CMAKE_BUILD_TYPE STREQUAL "Release")
     add_executable(bench_query benchmarks/bench_query_latency.cpp)
     target_link_libraries(bench_query edgevdb_core)
   endif()

4. android/CMakeLists.txt:
   cmake_minimum_required(VERSION 3.22)
   add_library(edgevdb SHARED
     src/main/cpp/vectordb_jni.cpp
     ${EDGEVDB_CORE_SOURCES}   # glob from ../core/src/
   )
   target_include_directories(edgevdb PRIVATE ../core/include ../core/vendor)
   target_compile_options(edgevdb PRIVATE -O2 -fvisibility=hidden)
   find_library(log-lib log)
   find_library(android-lib android)
   target_link_libraries(edgevdb ${log-lib} ${android-lib})
   # ONNX Runtime
   target_include_directories(edgevdb PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
   target_link_libraries(edgevdb ${ONNXRUNTIME_LIB})

5. CMakePresets.json — complete presets:
   {
     "version": 3,
     "configurePresets": [
       {"name":"desktop-debug","generator":"Ninja","binaryDir":"build/desktop-debug",
        "cacheVariables":{"CMAKE_BUILD_TYPE":"Debug"}},
       {"name":"desktop-release","generator":"Ninja","binaryDir":"build/desktop-release",
        "cacheVariables":{"CMAKE_BUILD_TYPE":"Release"}},
       {"name":"android-arm64","generator":"Ninja","binaryDir":"build/android-arm64",
        "toolchainFile":"${ANDROID_NDK}/build/cmake/android.toolchain.cmake",
        "cacheVariables":{"ANDROID_ABI":"arm64-v8a","ANDROID_PLATFORM":"android-26"}},
       {"name":"android-x86_64","generator":"Ninja","binaryDir":"build/android-x86_64",
        "toolchainFile":"${ANDROID_NDK}/build/cmake/android.toolchain.cmake",
        "cacheVariables":{"ANDROID_ABI":"x86_64","ANDROID_PLATFORM":"android-26"}}
     ],
     "buildPresets": [
       {"name":"desktop-debug","configurePreset":"desktop-debug"},
       {"name":"desktop-release","configurePreset":"desktop-release"}
     ]
   }
```

---

## Phase 13 — Testing & Benchmarks

### Prompt 13.1 — End-to-End Integration Tests

```
Create tests/test_e2e_rag.cpp — a full end-to-end integration test using only the C API.

This test simulates real usage:

TEST_CASE("End-to-end RAG pipeline") {
  // Setup: create temp directory, open database
  // Use evdb_default_config, override storage_dir to temp path
  
  SUBCASE("Insert 100 text chunks and query") {
    // 1. Open EdgeVDB
    // 2. Create Embedder from TEST_MODEL_PATH (skip if env var not set)
    // 3. Insert 100 synthetic chunks: 5 documents * 20 pages each
    //    Each chunk text: "Document {d} page {p}: {random_sentence}"
    // 4. Save index
    // 5. Query "page 5 document 3" → verify top result is from doc 3 page 5
    // 6. Verify context string is non-empty and within token budget
  }
  
  SUBCASE("Object store + relations") {
    // 1. Put 10 "Author" objects with {name, affiliation}
    // 2. Put 50 "Document" objects with {title, year}
    // 3. Add "authored_by" relations: each doc → one author
    // 4. Query objects by author name
    // 5. Traverse relations
    // 6. Save and reload — verify all data persists
  }
  
  SUBCASE("Sync round-trip") {
    // 1. Open two EdgeVDB instances in different temp dirs
    // 2. Insert 50 objects in instance A
    // 3. Insert 30 different objects in instance B
    // 4. Export delta from A, apply to B
    // 5. Export delta from B, apply to A
    // 6. Verify both instances have 80 unique objects
  }
  
  SUBCASE("Knowledge graph expansion") {
    // Insert chunks about "BERT" and "Transformers"
    // Insert chunks about "Attention mechanism" mentioning both
    // Query "BERT" with KG expansion enabled
    // Verify "Attention mechanism" chunks appear in expansion
  }
}
```

### Prompt 13.2 — Benchmarks

```
Create tests/benchmarks/bench_query_latency.cpp.

Benchmark suite measuring performance on the target hardware.
Results should be printed in a machine-parseable format: CSV with columns:
  platform, corpus_size, operation, median_ms, p95_ms, p99_ms

Benchmark scenarios:
1. Index build time:
   - Insert 1k, 10k, 50k, 100k random 384-dim vectors
   - Measure: total wall time in ms, time per chunk

2. Query latency (hybrid ranker, no KG):
   - Build index of 1k / 10k / 50k / 100k chunks
   - Run 100 queries (with 10-query warmup), measure each
   - Report median and p95

3. Query latency (with KG expansion):
   - Same as above but with KG expansion enabled
   - Report added latency from KG hop

4. Embedding throughput:
   - Embed 1000 sentences, measure total time and per-sentence time
   - Report tokens/sec estimate

5. Memory footprint:
   - Report peak RSS after building index at each corpus size
   - On Linux/Android: read /proc/self/status VmRSS
   - On macOS: use getrusage ru_maxrss

Target assertions (fail benchmark if missed in Release build):
- Query latency median < 100ms at 10k chunks on any platform
- Embedding per-sentence < 20ms on any platform
- Index build throughput > 10 chunks/sec on any platform
```

---

## Phase 14 — Documentation & Packaging

### Prompt 14.1 — Documentation

```
Create the following documentation files:

1. docs/architecture.md:
   - Full component diagram (ASCII art, same style as this guide)
   - Data flow diagrams for: insert, query, sync
   - Memory model: what is heap-allocated, what is mmap'd
   - Thread safety model: which components use which locks
   - File format reference: all .bin file layouts

2. docs/api_reference.md:
   - Auto-generated from vectordb.h Doxygen comments
   - Every function: description, parameters, return value, error conditions
   - Code example for each function

3. docs/android_integration.md:
   Step-by-step guide:
   a. Add edgevdb AAR to your Android project (Gradle)
   b. Place model.onnx and vocab.txt in app/src/main/assets/
   c. Initialize EdgeVDB in Application.onCreate()
   d. Use VectorDBManager from a ViewModel
   e. Complete code sample: PDF ingestion + query
   f. ProGuard rules to keep JNI class names
   g. APK size impact analysis

4. docs/ios_integration.md:
   Step-by-step guide:
   a. Add EdgeVDB Swift Package
   b. Bundle model.onnx and vocab.txt in app target
   c. Initialize in AppDelegate or Scene lifecycle
   d. Complete SwiftUI example: document import + query
   e. On-device privacy disclosure text

5. docs/python_integration.md:
   Step-by-step guide:
   a. Install: pip install edgevdb
   b. Download models: python -m edgevdb.download_models
   c. Complete example: folder of .txt files → indexed → queried
   d. Raspberry Pi notes: expected latency, memory usage

6. README.md (root):
   - One-line description
   - Feature badge table
   - Quick-start for each platform (10-line code snippet each)
   - Performance table (target numbers with XXX placeholders to fill after benchmarks)
   - Architecture diagram
   - License section
   - Link to each docs/*.md
```

### Prompt 14.2 — Model Download Script

```
Create models/download_models.sh:
#!/bin/bash
# Downloads and quantizes all-MiniLM-L6-v2 ONNX model for EdgeVDB
# Produces: model.onnx (INT8 quantized), vocab.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR}"

echo "Downloading all-MiniLM-L6-v2..."
pip install -q optimum[onnxruntime] transformers

python3 - <<'PYTHON'
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import shutil, os

model_id = "sentence-transformers/all-MiniLM-L6-v2"
print("Exporting to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("/tmp/minilm_onnx")
tokenizer.save_pretrained("/tmp/minilm_onnx")

# Copy model and vocab
import shutil
shutil.copy("/tmp/minilm_onnx/model.onnx", os.environ.get("OUTPUT_DIR","models") + "/model.onnx")

# Extract vocab.txt
vocab_file = "/tmp/minilm_onnx/vocab.txt"
if os.path.exists(vocab_file):
    shutil.copy(vocab_file, os.environ.get("OUTPUT_DIR","models") + "/vocab.txt")
    
print("Done! model.onnx and vocab.txt are ready.")
PYTHON

Also create python/edgevdb/download_models.py:
A Python script (callable via python -m edgevdb.download_models) that does the same
programmatically, accepting --output-dir argument.
```

### Prompt 14.3 — CI/CD Configuration

```
Create .github/workflows/ with the following GitHub Actions workflows:

1. ci.yml — Runs on every PR and push to main:
   Jobs:
   a. build-desktop:
      - matrix: [ubuntu-latest, macos-latest, windows-latest]
      - Install Ninja, CMake 3.22+
      - cmake --preset desktop-debug
      - cmake --build --preset desktop-debug
      - ctest --test-dir build/desktop-debug (skip integration tests)
   
   b. build-android:
      - runs-on: ubuntu-latest
      - Uses action: android-actions/setup-android@v3
      - Install NDK 27.x
      - cmake --preset android-arm64
      - cmake --build --preset android-arm64
      - (No test run — requires device)
   
   c. lint-cpp:
      - clang-tidy on all core/src/*.cpp
      - clang-format --dry-run --Werror check

2. release.yml — Runs on tag push v*:
   a. Build all platforms
   b. Run full test suite
   c. Package:
      - edgevdb-android-{version}.aar
      - EdgeVDB-ios-{version}.xcframework.zip
      - edgevdb-python-{version}-{platform}.whl
   d. Create GitHub Release with artifacts attached
   e. Publish Python wheel to PyPI (using Trusted Publisher)

3. benchmark.yml — Runs weekly on main:
   - Runs bench_query_latency on ubuntu-latest
   - Posts results as PR comment or to repo wiki
```

---

## Why Relational DB & Sync Were Missing — Design Rationale

Understanding why EdgeVDB paper omitted these features, and why your SDK must add them:

### Why the EdgeVDB paper omitted them

| Missing feature | Why omitted in paper |
|---|---|
| **Relational Object Store** | The paper's scope is purely *retrieval quality and latency*. The RAG pipeline only needs vectors and chunk text — no structured object metadata was required to prove the HNSW + hybrid ranker novelty claims. |
| **Data Sync** | Sync is architecturally orthogonal to single-device RAG. It requires a wire protocol, conflict resolution policy, and either a relay server or P2P transport — all beyond what a retrieval quality paper covers. |

### Why your SDK must include them to compete

ObjectBox's competitive advantage is precisely these two features on top of vector search. Without them, developers must stitch together EdgeVDB + SQLite + a custom sync solution. By bundling all three in one zero-dependency library, your SDK becomes a complete edge AI data stack:

```
Vector DB (HNSW)  →  "Find semantically similar chunks"
Object Store      →  "Store structured metadata about documents, users, sessions"
Relations         →  "Link chunks to their source documents, authors, categories"
Data Sync         →  "Sync a user's personal corpus across their phone, tablet, PC"
```

### Architecture of the relational store

The ObjectRecord approach (msgpack blob + type/property index) is intentionally simpler than a full SQL engine:
- No schema migrations (schema-less, properties are dynamic JSON)
- No JOIN operations (use explicit relations via RelationIndex)
- No transactions (LWW sync handles conflicts)
- This trades SQL power for zero-dependency portability across all target platforms

### Architecture of the sync engine

The CRDT (Vector Clock + LWW) approach was chosen over OT (Operational Transform) because:
- No central server required for conflict resolution
- Correctness proof is simple: higher clock always wins
- Works offline-first: sync when connection available
- Scales to any number of devices without coordination

The only limitation: two devices editing the same object field simultaneously will lose one edit. For RAG applications where objects are mostly append-only (new documents, new chunks), this is acceptable.

---

## Full Feature Matrix

Final deliverable feature check — the IDE must verify each is implemented:

| Feature | C++ Module | Android API | iOS API | Python API |
|---|---|---|---|---|
| HNSW ANN index | hnsw_index.cpp | ✓ | ✓ | ✓ |
| Page proximity signal | page_index.cpp + hybrid_ranker.cpp | ✓ | ✓ | ✓ |
| Keyword Jaccard overlap | hybrid_ranker.cpp | ✓ | ✓ | ✓ |
| Configurable ranker weights | hybrid_ranker.cpp | ✓ | ✓ | ✓ |
| KG entity extraction | kg_extractor.cpp | ✓ | ✓ | ✓ |
| KG multi-hop expansion | kg_expander.cpp | ✓ | ✓ | ✓ |
| Text → embedding pipeline | tokenizer.cpp + embedder.cpp | ✓ | ✓ | ✓ |
| ONNX Runtime (MiniLM) | embedder.cpp | ✓ (NNAPI) | ✓ | ✓ |
| ARM NEON SIMD dot product | hnsw_index.cpp | ✓ | ✓ | ✓ |
| Token budget enforcement | token_budget.cpp + query_engine.cpp | ✓ | ✓ | ✓ |
| Relational object store | object_store.cpp | ✓ | ✓ | ✓ |
| Typed property indexing | object_store.cpp | ✓ | ✓ | ✓ |
| Relation edges (FK) | relation_index.cpp | ✓ | ✓ | ✓ |
| Combined vector+object query | query_engine.cpp | ✓ | ✓ | ✓ |
| CRDT sync (LWW) | sync_engine.cpp | ✓ | ✓ | ✓ |
| File-based delta export/import | sync_engine.cpp | ✓ | ✓ | ✓ |
| Binary serialization + CRC32 | all store modules | ✓ | ✓ | ✓ |
| Stable C public API | vectordb.h | ✓ | ✓ | ✓ |
| JNI bridge | vectordb_jni.cpp | ✓ | — | — |
| Swift wrapper | — | — | ✓ | — |
| ctypes Python wrapper | — | — | — | ✓ |
| CMake unified build | CMakeLists.txt | ✓ | ✓ | ✓ |
| Doctest C++ unit tests | tests/ | — | — | — |
| pytest Python tests | python/tests/ | — | — | ✓ |
| GitHub Actions CI | .github/workflows/ | ✓ | ✓ | ✓ |
| Model download script | models/ | ✓ | ✓ | ✓ |
| Full documentation | docs/ | ✓ | ✓ | ✓ |

---

## Implementation Order Summary

Execute the phases in this exact order to avoid dependency issues:

```
Phase 0  → Scaffolding & toolchain
Phase 1  → Schema + ChunkStore + PageIndex (foundation data layer)
Phase 2  → HNSW Index (depends on schema)
Phase 3  → Hybrid Ranker (depends on HNSW + PageIndex + ChunkStore)
Phase 4  → Knowledge Graph (depends on ChunkStore)
Phase 5  → Tokenizer + ONNX Embedder (independent, but needed for E2E)
Phase 6  → Object Store + Relations + Query Engine (independent data layer)
Phase 7  → Sync Engine (depends on Object Store + ChunkStore)
Phase 8  → C API Facade (depends on all of phases 1–7)
Phase 9  → Android JNI + Kotlin (depends on Phase 8)
Phase 10 → iOS Swift (depends on Phase 8)
Phase 11 → Python ctypes (depends on Phase 8)
Phase 12 → CMake build system (refine throughout, finalize here)
Phase 13 → Tests + Benchmarks (depends on all phases)
Phase 14 → Docs + packaging (final)
```

**Total estimated source files:** ~60 C++ files, ~15 Kotlin files, ~8 Swift files, ~8 Python files  
**Total estimated lines of code:** ~25,000–35,000 lines  
**Build time (Release, desktop):** ~3–5 minutes  
**Final .so size (Android arm64, stripped):** target < 4 MB
```
