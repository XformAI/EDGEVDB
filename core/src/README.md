# EdgeVDB Core — Implementation Source

> **C++17 implementation of the EdgeVDB vector database core with HNSW ANN, hybrid retrieval, knowledge graph, and CRDT sync.**

This directory contains the complete C++17 implementation of the EdgeVDB core library. All source files are designed for:

- **Zero external dependencies** — Only standard library and vendored headers
- **Cross-platform** — Linux, macOS, Windows, Android, iOS, Raspberry Pi
- **Performance** — SIMD optimizations (NEON/SSE2) for vector operations
- **Thread safety** — Shared mutexes for concurrent access
- **Persistence** — Binary serialization with CRC32 checksums

## Source Files

### Core Database

#### `vectordb_impl.cpp`
Database implementation orchestrator that ties together all components.

**Responsibilities:**
- Initialize and coordinate all subsystems (HNSW, chunk store, object store, etc.)
- Implement the main database lifecycle (open/save/close)
- Coordinate between vector store and object store
- Handle configuration and initialization

**Key Classes:**
- `VectorDatabase` — Main database class integrating all components

#### `vectordb_c_api.cpp`
C API implementation that wraps C++ classes for ABI stability.

**Responsibilities:**
- Implement all C API functions from `vectordb.h`
- Convert between C handles and C++ objects
- Handle error code translation
- Manage opaque handle lifecycle

**Key Functions:**
- `evdb_open()` → `VectorDatabase` construction
- `evdb_insert_chunk()` → Chunk store insertion
- `evdb_query_vector()` → HNSW search + hybrid ranking
- All other C API functions

### Vector Index

#### `hnsw_index.hpp` / `hnsw_index.cpp`
Hierarchical Navigable Small World (HNSW) ANN index implementation.

**Algorithm:**
- Multi-layer graph structure for fast approximate nearest neighbor search
- M=16 connections per node, ef_construction=200, ef_search=64
- Cosine distance for L2-normalized vectors (384 dimensions)
- Dynamic graph construction with probabilistic layer assignment

**Key Features:**
- SIMD-optimized distance computation (NEON for ARM, SSE2 for x86)
- Thread-safe with shared mutex
- Persistent storage with binary serialization
- CRC32 checksums for data integrity

**Key Classes:**
- `HNSWIndex` — Main index class
- `HNSWNode` — Graph node structure
- `DistPair` — (distance, internal_id) tuple

**Performance:**
- Insert: O(log N) average
- Query: O(log N) average with configurable ef
- Memory: ~O(N) for N vectors

**SIMD Optimizations:**
```cpp
// ARM NEON (arm64-v8a)
float32x4_t sum = vmlaq_f32(sum, va, vb);

// x86 SSE2
__m128 sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));

// Scalar fallback (8x unrolled)
for (size_t i = 0; i + 8 <= dim; i += 8) {
    s0 += a[i] * b[i];
    // ...
}
```

### Chunk Storage

#### `chunk_store.hpp` / `chunk_store.cpp`
Persistent storage for text chunks with metadata.

**Data Model:**
```cpp
struct ChunkNode {
    uint64_t id;
    uint32_t doc_id;
    uint32_t page_number;
    char text[MAX_TEXT_LEN];
    // ... other fields
};
```

**Features:**
- In-memory hash map for fast lookup (O(1) average)
- Binary serialization to disk
- CRC32 checksums for integrity
- Soft delete support
- Thread-safe with shared mutex

**File Format:**
- Magic bytes: "EVDBCHK\0"
- Version: uint32_t
- Chunk count: uint64_t
- Chunk records (variable length)

### Page Index

#### `page_index.hpp` / `page_index.cpp`
Page proximity index for hybrid ranking.

**Purpose:**
- Track which chunks belong to which pages
- Compute page proximity scores for re-ranking
- Support page-aware retrieval

**Key Methods:**
- `addChunk(chunk_id, page_number)` — Index chunk by page
- `computePageProximity(chunk_a, chunk_b)` — Calculate proximity score
- `getChunksForPage(page_number)` — Get all chunks on a page

**Scoring:**
- Same page: 1.0
- Adjacent pages: 0.8
- Nearby pages: 0.5
- Distant pages: 0.1

### Hybrid Ranking

#### `hybrid_ranker.hpp` / `hybrid_ranker.cpp`
Multi-factor ranking combining vector similarity, page proximity, and keyword overlap.

**Algorithm:**
```
final_score = α * cosine_score + β * page_score + γ * keyword_score
```

**Default Weights:**
- α (cosine): 0.70
- β (page): 0.20
- γ (keyword): 0.10

**Keyword Overlap:**
- Tokenizes query and chunk text
- Removes stopwords (50 common English words)
- Computes Jaccard coefficient
- Case-insensitive matching

**Key Classes:**
- `HybridRanker` — Main ranking engine
- `RankerInput` — Input structure with HNSW results and context

### Knowledge Graph

#### `knowledge_graph.hpp` / `knowledge_graph.cpp`
Entity graph for multi-hop knowledge expansion.

**Data Model:**
- Entities extracted from text via NER
- Entity-to-chunks mapping (many-to-many)
- Entity co-occurrence graph

**Key Methods:**
- `addChunkEntities(chunk_id, entities)` — Index entities for a chunk
- `getChunksForEntity(entity)` — Get chunks mentioning an entity
- `getRelatedEntities(entity, hops)` — Multi-hop graph traversal

**Graph Structure:**
```cpp
entity_to_chunks_:      // entity → [chunk_ids]
entity_cooccurrence_:   // entity → [related_entities]
```

#### `kg_extractor.hpp` / `kg_extractor.cpp`
Named Entity Recognition (NER) for knowledge graph construction.

**Implementation:**
- Rule-based NER with pattern matching
- Entity types: PERSON, ORGANIZATION, LOCATION, DATE, NUMBER
- Capitalization and pattern heuristics
- No ML dependencies for zero-dependency requirement

**Entity Types:**
- **PERSON**: Names (e.g., "John Smith")
- **ORGANIZATION**: Companies, institutions (e.g., "Google")
- **LOCATION**: Places (e.g., "New York")
- **DATE**: Temporal expressions (e.g., "January 2024")
- **NUMBER**: Numeric values (e.g., "42")

#### `kg_expander.hpp` / `kg_expander.cpp`
Multi-hop entity expansion for query enhancement.

**Purpose:**
- Expand queries with related entities
- Improve recall for entity-based queries
- Support knowledge graph traversal

**Algorithm:**
1. Extract entities from query
2. Look up related entities (1-3 hops)
3. Generate expanded query terms
4. Boost chunks containing related entities

### Object Store

#### `object_store.hpp` / `object_store.cpp`
Schema-less NoSQL document store with indexing.

**Data Model:**
```cpp
struct ObjectRecord {
    uint64_t id;
    std::string type_name;
    nlohmann::json properties;
    uint64_t timestamp;
    // ... sync fields
};
```

**Features:**
- JSON property storage via nlohmann/json
- Type-based indexing (type → object IDs)
- Property-based indexing (type, property, value → object IDs)
- Soft delete support
- Thread-safe with shared mutex

**Indexes:**
- Primary index: `id → ObjectRecord`
- Type index: `type_name → [object_ids]`
- Property index: `(type, property, value) → [object_ids]`

**Query Types:**
- Get by ID
- Get all by type
- Query by property value
- Pagination support

#### `relation_index.hpp` / `relation_index.cpp`
Typed relation index for object graph.

**Data Model:**
```cpp
struct RelationEdge {
    std::string relation_type;
    uint64_t from_id;
    uint64_t to_id;
    uint64_t timestamp;
    // ... sync fields
};
```

**Features:**
- Directed edges with typed relations
- Forward and reverse indexing
- Multi-hop traversal support
- CRDT sync support

**Indexes:**
- Forward: `(from_id, relation_type) → [to_ids]`
- Reverse: `(to_id, relation_type) → [from_ids]`

### Embedding Pipeline

#### `embedder.hpp` / `embedder.cpp`
Text embedding using ONNX Runtime.

**Model:**
- all-MiniLM-L6-v2 (sentence-transformers)
- 384-dimensional output
- WordPiece tokenization
- Mean pooling over tokens

**Features:**
- ONNX Runtime integration (optional)
- Fallback hash embedder for zero-dependency mode
- Thread-safe for concurrent embedding
- L2 normalization for cosine similarity

**Fallback Hash Embedder:**
- Deterministic hash-based embedding
- No ONNX dependency
- Useful for testing and prototyping
- Not production-quality for semantic search

#### `tokenizer.hpp` / `tokenizer.cpp`
WordPiece tokenizer for BERT-based models.

**Implementation:**
- WordPiece subword tokenization
- Vocabulary lookup (30,522 tokens)
- Special tokens: [CLS], [SEP], [PAD], [UNK]
- Maximum sequence length: 512 tokens

**Tokenization Pipeline:**
1. Basic tokenization (lowercase, split on whitespace/punctuation)
2. WordPiece subword tokenization (greedy longest match)
3. Add special tokens ([CLS], [SEP])
4. Pad/truncate to max length
5. Generate attention mask

### Query Engine

#### `query_engine.hpp` / `query_engine.cpp`
Query orchestration combining HNSW search and hybrid ranking.

**Query Pipeline:**
1. HNSW ANN search (vector similarity)
2. Page proximity scoring
3. Keyword overlap scoring
4. Hybrid ranking (weighted combination)
5. Knowledge graph expansion (optional)
6. Token budget assembly (optional)

**Key Classes:**
- `QueryEngine` — Main query orchestrator
- `QueryResult` — Ranked result with scores

### Sync Engine

#### `sync_engine.hpp` / `sync_engine.cpp`
CRDT-based synchronization engine for multi-device sync.

**Algorithm:**
- Last-Writer-Wins (LWW) with vector clocks
- Per-device logical clocks
- Delta-based synchronization
- Conflict resolution by timestamp

**Data Model:**
```cpp
struct DeviceClock {
    std::string device_id;
    uint64_t logical_clock;
};

struct SyncDelta {
    std::string source_device_id;
    uint64_t from_clock;
    uint64_t to_clock;
    std::vector<ObjectRecord> records;
    std::vector<RelationEdge> edges;
    std::vector<ChunkNode> chunks;
};
```

**Operations:**
- `exportDelta(since_clock)` — Export changes since clock
- `applyDelta(delta)` — Apply received delta
- `exportToFile(path, since_clock)` — Export to file
- `importFromFile(path)` — Import from file

**Conflict Resolution:**
- LWW based on timestamp
- No manual conflict resolution
- Automatic merge with deterministic outcome

#### `sync_protocol.hpp`
Sync protocol definitions and serialization.

**Contents:**
- SyncDelta structure
- Serialization/deserialization methods
- Protocol version information

### Token Budget

#### `token_budget.hpp` / `token_budget.cpp`
Token budget management for RAG context assembly.

**Purpose:**
- Limit context size for LLM prompts
- Prioritize high-relevance chunks
- Approximate token counting (word-based)

**Algorithm:**
1. Sort chunks by relevance score
2. Add chunks until token budget exceeded
3. Truncate final chunk if needed
4. Assemble context string

**Approximation:**
- 1 token ≈ 1.3 words (English)
- Configurable budget (default: 3200 tokens)
- Word-level counting for speed

### Schema

#### `schema.hpp`
Common data structures and type definitions.

**Key Structures:**
- `ChunkNode` — Chunk data model
- `ObjectRecord` — Object data model
- `RelationEdge` — Relation data model
- `Entity` — Knowledge graph entity
- `QueryResult` — Query result with scores

**Constants:**
- `MAX_TEXT_LEN` — Maximum chunk text length (2048 chars)
- `EMBEDDING_DIM` — Embedding dimensions (384)
- Magic bytes for file headers

### Logging

#### `log.hpp`
Simple logging utility.

**Levels:**
- 0: Off
- 1: Error only
- 2: Info
- 3: Debug

**Usage:**
```cpp
LOG_ERROR("Failed to open database: %s", path.c_str());
LOG_INFO("Inserted chunk: %llu", chunk_id);
LOG_DEBUG("HNSW search: ef=%d, k=%d", ef, k);
```

## Build Configuration

### SIMD Detection

Automatic SIMD detection based on platform:
```cpp
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define EDGEVDB_USE_NEON 1
#elif defined(__SSE2__) || defined(_M_X64)
#define EDGEVDB_USE_SSE2 1
#endif
```

### Compiler Flags

Required flags:
- `-std=c++17` — C++17 standard
- `-Wall -Wextra -Wpedantic` — Strict warnings
- `-fvisibility=hidden` — Hide symbols by default
- `-fstack-protector-strong` — Stack protection
- `-D_FORTIFY_SOURCE=2` — Runtime hardening

Platform-specific:
- ARM: `-march=armv8-a+simd` (NEON)
- x86_64: `-msse4.2` (SSE2)

## Performance Characteristics

### HNSW Index

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert | O(log N) | Average case |
| Query | O(log N) | With ef parameter |
| Delete | O(log N) | Soft delete |
| Memory | O(N) | Linear in vector count |

### Object Store

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert | O(1) | Hash map insert |
| Get by ID | O(1) | Hash map lookup |
| Query by type | O(k) | k = objects of type |
| Query by property | O(k) | k = matching objects |

### Knowledge Graph

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Add entities | O(e) | e = entities per chunk |
| Get chunks for entity | O(k) | k = chunks with entity |
| Multi-hop expansion | O(h·d) | h = hops, d = degree |

### Memory Footprint

| Component | Per-Item Overhead | Notes |
|-----------|------------------|-------|
| HNSW Node | ~200 bytes | Depends on M parameter |
| Chunk | ~2.5 KB | Includes text and metadata |
| Object | ~500 bytes | JSON properties |
| Relation | ~100 bytes | Edge metadata |

## Thread Safety

### Locking Strategy

- **Shared mutex** for read-heavy workloads
- **Read locks** for queries (concurrent reads allowed)
- **Write locks** for mutations (exclusive access)

### Thread-Safe Components

- `HNSWIndex` — Shared mutex
- `ChunkStore` — Shared mutex
- `ObjectStore` — Shared mutex
- `RelationIndex` — Shared mutex
- `KnowledgeGraph` — Shared mutex

### Non-Thread-Safe Components

- `Embedder` — Thread-safe for concurrent `embed()` calls
- `QueryEngine` — Use per-thread instances
- `SyncEngine` — External locking required

## Persistence

### File Format

All binary files use:
- Magic bytes (8 chars) for identification
- Version number (uint32_t)
- CRC32 checksums for integrity
- Little-endian byte order

### File Locations

Default file names:
- `chunks.bin` — Chunk store
- `objects.bin` — Object store
- `relations.bin` — Relation index
- `hnsw.bin` — HNSW index
- `kg.bin` — Knowledge graph
- `sync.bin` — Sync state

### Serialization

Binary serialization using:
- Fixed-size types (uint32_t, uint64_t, float)
- Length-prefixed strings
- JSON for complex structures (via nlohmann/json)
- CRC32 for checksums

## Testing

### Unit Tests

Located in `../../tests/`:
- `test_hnsw.cpp` — HNSW index tests
- `test_hybrid_ranker.cpp` — Ranking algorithm tests
- `test_embedder.cpp` — Embedding pipeline tests
- `test_object_store.cpp` — Object store tests
- `test_sync.cpp` — Sync engine tests
- `test_e2e_rag.cpp` — End-to-end RAG pipeline tests

### Benchmarks

Located in `../../tests/benchmarks/`:
- `bench_query_latency.cpp` — Query latency at scale
- `bench_build_time.cpp` — Index build throughput

## Contributing

### Adding New Components

1. Create header file in `src/`
2. Create implementation file in `src/`
3. Add to `CMakeLists.txt`
4. Add unit tests in `../../tests/`
5. Update this README

### Code Style

- Follow Google C++ Style Guide
- Use `clang-format` for formatting
- Add comments for non-obvious logic
- Use `const` and `constexpr` where possible
- Prefer RAII for resource management

### Performance Guidelines

- Profile before optimizing
- Use SIMD for vector operations
- Minimize allocations in hot paths
- Use move semantics for large objects
- Cache frequently accessed data

## See Also

- [../include/README.md](../include/README.md) — Public API headers
- [../CMakeLists.txt](../CMakeLists.txt) — Build configuration
- [../../README.md](../../README.md) — Project overview
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
