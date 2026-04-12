# EdgeVDB Architecture

> **Comprehensive architecture documentation for the EdgeVDB on-device vector database.**

## System Overview

EdgeVDB is a cross-platform, embeddable vector database designed for on-device RAG (Retrieval-Augmented Generation). It combines HNSW approximate nearest neighbor search with a hybrid re-ranking pipeline, knowledge graph expansion, and a relational object store — all with zero required dependencies.

**Key Design Principles:**
- **Zero Dependencies** — Core library requires no external libraries
- **ONNX Optional** — Use pre-computed embeddings from any provider
- **Cross-Platform** — Linux, macOS, Windows, Android, iOS, Raspberry Pi
- **Performance** — SIMD optimizations (NEON/SSE2) for vector operations
- **Privacy** — All data remains on-device, no external API calls
- **Simplicity** — Stable C API for easy FFI bindings

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Public C API (vectordb.h)                    │
│                  Stable ABI, Opaque Handles                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                       Query Engine                          │ │
│  │  Text → Embed → HNSW ANN → Hybrid Rerank → Token Budget    │ │
│  │                    ↕            ↕                            │ │
│  │              Page Index    KG Expander                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Chunk Store  │  │  HNSW Index  │  │  Knowledge Graph     │   │
│  │  (flat file)  │  │  (float32)   │  │  (entity→chunk map)  │   │
│  │  - CRC32      │  │  - NEON/SSE2 │  │  - Multi-hop         │   │
│  │  - mmap       │  │  - L2 norm   │  │  - NER              │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Object Store  │  │  Relations   │  │   Sync Engine        │   │
│  │ (NoSQL)       │  │  (FK edges)  │  │   (CRDT / LWW)      │   │
│  │ - JSON        │  │  - Typed     │  │  - Vector Clock      │   │
│  │ - Indexing    │  │  - Directed  │  │  - Delta Sync        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Embedder (optional): WordPiece Tokenizer → ONNX MiniLM │    │
│  │  - 384-dim output                                        │    │
│  │  - L2 normalization                                     │    │
│  │  - Mean pooling                                         │    │
│  │  Or: use pre-computed embeddings via insert_chunk()      │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Core Components

### HNSW Index

Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index.

**Algorithm:**
- Multi-layer graph structure for logarithmic search complexity
- Probabilistic layer assignment (exponential distribution)
- Dynamic graph construction during insertion
- Beam search for query time

**Parameters:**
- `M` (default: 16): Maximum connections per node
- `ef_construction` (default: 200): Search width during construction
- `ef_search` (default: 64): Search width during query

**Performance:**
- Insert: O(log N) average
- Query: O(log N) average
- Memory: O(N) for N vectors

**SIMD Optimizations:**
- ARM NEON for arm64-v8a
- SSE2 for x86_64
- Scalar fallback with 8x unrolling

### Chunk Store

Persistent storage for text chunks with metadata.

**Data Model:**
```cpp
struct ChunkNode {
    uint64_t id;
    uint32_t doc_id;
    uint32_t page_number;
    char text[MAX_TEXT_LEN];  // 2048 chars
    float embedding[384];      // Optional for rebuild
    uint64_t timestamp;
    uint32_t crc32;
};
```

**Features:**
- In-memory hash map for O(1) lookup
- Binary serialization to disk
- CRC32 checksums for integrity
- Soft delete support
- Thread-safe with shared mutex

### Page Index

Page proximity index for hybrid ranking.

**Purpose:**
- Track which chunks belong to which pages
- Compute page proximity scores for re-ranking
- Support page-aware retrieval

**Scoring:**
- Same page: 1.0
- Adjacent pages: 0.8
- Nearby pages: 0.5
- Distant pages: 0.1

### Hybrid Ranker

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

### Knowledge Graph

Entity graph for multi-hop knowledge expansion.

**Components:**
- **KGExtractor**: Named Entity Recognition (rule-based)
- **KnowledgeGraph**: Entity-to-chunks mapping and graph traversal
- **KGExpander**: Multi-hop query expansion

**Entity Types:**
- PERSON: Names (e.g., "John Smith")
- ORGANIZATION: Companies, institutions (e.g., "Google")
- LOCATION: Places (e.g., "New York")
- DATE: Temporal expressions (e.g., "January 2024")
- NUMBER: Numeric values (e.g., "42")

**Graph Structure:**
- `entity_to_chunks`: entity → [chunk_ids]
- `entity_cooccurrence`: entity → [related_entities]

### Object Store

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

**Indexes:**
- Primary: `id → ObjectRecord`
- Type: `type_name → [object_ids]`
- Property: `(type, property, value) → [object_ids]`

**Features:**
- JSON property storage via nlohmann/json
- Type-based and property-based indexing
- Soft delete support
- Thread-safe with shared mutex

### Relation Index

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

**Indexes:**
- Forward: `(from_id, relation_type) → [to_ids]`
- Reverse: `(to_id, relation_type) → [from_ids]`

### Sync Engine

CRDT-based synchronization engine for multi-device sync.

**Algorithm:**
- Last-Writer-Wins (LWW) with vector clocks
- Per-device logical clocks
- Delta-based synchronization
- Automatic conflict resolution

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

### Embedder

Text embedding using ONNX Runtime (optional).

**Model:**
- all-MiniLM-L6-v2 (sentence-transformers)
- 384-dimensional output
- WordPiece tokenization
- Mean pooling over tokens

**Fallback:**
- Hash-based embedder for zero-dependency mode
- Not production-quality for semantic search
- Useful for testing and prototyping

### Query Engine

Query orchestration combining all components.

**Pipeline:**
1. Embed query (if using auto-embedding)
2. HNSW ANN search (vector similarity)
3. Page proximity scoring
4. Keyword overlap scoring
5. Hybrid ranking (weighted combination)
6. Knowledge graph expansion (optional)
7. Token budget assembly (optional)

### Token Budget

Token budget management for RAG context assembly.

**Purpose:**
- Limit context size for LLM prompts
- Prioritize high-relevance chunks
- Approximate token counting (word-based)

**Algorithm:**
- Sort chunks by relevance score
- Add chunks until token budget exceeded
- Truncate final chunk if needed
- Assemble context string

## Data Flow

### Insert (with pre-computed embedding)

1. **Caller provides** text + 384-dim float32 embedding
2. **ChunkStore.put()** assigns ID and timestamp
3. **HNSWIndex.insert()** builds graph connections
4. **PageIndex.insert()** maps chunk to doc/page
5. **KGExtractor.extract()** performs NER
6. **KnowledgeGraph.addChunkEntities()** creates entity-chunk mapping
7. **Save to disk** (optional, explicit save required)

### Insert (with auto-embedding)

1. **Text → WordPiece tokenizer** → token IDs
2. **Token IDs → ONNX Runtime inference** → 384-dim float32 embedding
3. **L2 normalize** embedding
4. **Then same steps** as pre-computed path (2–7 above)

### Query

1. **Query text → WordPiece tokenize** → ONNX inference → 384-dim embedding (via `query_text()`)
   - **OR** caller provides pre-computed embedding directly (via `query_vector()` — no ONNX cost)
2. **HNSW KNN search** with over-fetch (3× top_k)
3. **HybridRanker re-ranks**: α·cosine + β·page_proximity + γ·keyword
4. **Optional**: KG expansion adds related chunks via entity graph
5. **TokenBudget** trims to fit LLM context window
6. **Returns** `QueryResult[]` and assembled context string

## Binary File Formats

All stores use a consistent binary format:

**Structure:**
- 8-byte magic (store-specific, e.g., "EVDBCHK\0")
- 4-byte version (uint32)
- 8-byte record count (uint64)
- N × record (struct, trivially-copyable)
- 4-byte CRC32 trailer

**Magic Bytes:**
- Chunk Store: "EVDBCHK\0"
- HNSW Index: "EVDBHNSW"
- Object Store: "EVDBOBJ\0"
- Relations: "EVDBREL\0"
- Knowledge Graph: "EVDBKG\0"
- Sync: "EVDBSYNC\0"

**File Locations:**
Default file names in storage directory:
- `chunks.bin` — Chunk store
- `hnsw.bin` — HNSW index
- `objects.bin` — Object store
- `relations.bin` — Relation index
- `kg.bin` — Knowledge graph
- `sync.bin` — Sync state

## Concurrency Model

All stores use `std::shared_mutex` for readers-writer locking:

**Locking Strategy:**
- Multiple concurrent readers (shared lock)
- Exclusive writers (exclusive lock)
- Thread-safe from JNI/Swift/Python

**Thread-Safe Components:**
- HNSWIndex — Shared mutex
- ChunkStore — Shared mutex
- ObjectStore — Shared mutex
- RelationIndex — Shared mutex
- KnowledgeGraph — Shared mutex

**Non-Thread-Safe Components:**
- Embedder — Thread-safe for concurrent `embed()` calls
- QueryEngine — Use per-thread instances
- SyncEngine — External locking required

## Memory Management

**RAII Patterns:**
- C++ RAII wrappers for automatic cleanup
- C API requires explicit handle management
- Python SDK uses context managers
- Android SDK uses Kotlin coroutines with cleanup

**Handle Lifecycle:**
1. Create handle via `*_create` or `*_open`
2. Use handle for operations
3. Call `*_destroy` or `*_close` to release
4. Query handles must be freed before closing database

**Memory Footprint:**

| Component | Per-Item Overhead | Notes |
|-----------|------------------|-------|
| HNSW Node | ~200 bytes | Depends on M parameter |
| Chunk | ~2.5 KB | Includes text and metadata |
| Object | ~500 bytes | JSON properties |
| Relation | ~100 bytes | Edge metadata |

## Platform Support

| Platform | Integration | Language | SIMD |
|----------|-------------|----------|------|
| Android | JNI shared library | Kotlin | NEON |
| iOS | XCFramework / SPM | Swift | NEON |
| Desktop (Linux) | Shared library | Python (ctypes), C/C++ | SSE2 |
| Desktop (macOS) | Shared library | Python (ctypes), C/C++ | SSE2 |
| Desktop (Windows) | DLL | Python (ctypes), C/C++ | SSE2 |
| Raspberry Pi | Shared library | Python (ctypes), C/C++ | NEON |

**Platform-Specific Optimizations:**
- ARM64: NEON SIMD for vector operations
- x86_64: SSE2 SIMD for vector operations
- Scalar fallback for other architectures

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

## Performance Targets

| Metric | Target | Platform |
|--------|--------|----------|
| Query latency (10k chunks) | < 100ms | Desktop |
| Query latency (10k chunks) | < 200ms | Raspberry Pi |
| Embedding latency | < 50ms | Desktop |
| Index build (10k chunks) | < 5s | Desktop |
| Library size (stripped) | < 4 MB | Android arm64 |
| Memory footprint (10k chunks) | < 100 MB | Desktop |

## Security Considerations

### Data Privacy
- All data remains on-device
- No external API calls required
- No telemetry or analytics
- Offline operation supported

### Model Integrity
- CRC32 checksums for all files
- Magic bytes for format validation
- Version checking for compatibility

### Access Control
- File system permissions for database files
- No built-in encryption (use OS-level encryption)
- Recommended: Encrypted filesystems for sensitive data

## See Also

- [api_reference.md](api_reference.md) — Complete C API reference
- [../core/include/README.md](../core/include/README.md) — Public API headers
- [../core/src/README.md](../core/src/README.md) — Implementation details
- [../README.md](../README.md) — Project overview
