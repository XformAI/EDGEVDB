# EdgeVDB Architecture

## System Overview

EdgeVDB is a cross-platform, embeddable vector database designed for on-device RAG (Retrieval-Augmented Generation). It combines HNSW approximate nearest neighbor search with a hybrid re-ranking pipeline, knowledge graph expansion, and a relational object store.

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Public C API (vectordb.h)                    │
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
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Object Store  │  │  Relations   │  │   Sync Engine        │   │
│  │ (NoSQL)       │  │  (FK edges)  │  │   (CRDT / LWW)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Embedding Pipeline: WordPiece Tokenizer → ONNX MiniLM  │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Insert
1. Text → WordPiece tokenizer → token IDs
2. Token IDs → ONNX Runtime inference → 384-dim float32 embedding
3. L2 normalize embedding
4. ChunkStore.put() → assigns ID and timestamp
5. HNSWIndex.insert() → builds graph connections
6. PageIndex.insert() → maps chunk to doc/page
7. KGExtractor.extract() → NER entities
8. KnowledgeGraph.addChunkEntities() → entity-chunk mapping

### Query
1. Query text → embed (same pipeline as insert)
2. HNSW KNN search with over-fetch (3× top_k)
3. HybridRanker re-ranks: α·cosine + β·page_proximity + γ·keyword
4. Optional: KG expansion adds related chunks via entity graph
5. TokenBudget trims to fit LLM context window
6. Returns `QueryResult[]` and assembled context string

## Binary File Formats

All stores use a consistent format:
- 8-byte magic (store-specific)
- 4-byte version (uint32)
- 8-byte record count (uint64)
- N × record (struct, trivially-copyable)
- 4-byte CRC32 trailer

## Concurrency

All stores use `std::shared_mutex` for readers-writer locking:
- Multiple concurrent readers
- Exclusive writers
- Thread-safe from JNI/Swift/Python

## Platform Support

| Platform | Integration | Language |
|----------|-------------|----------|
| Android | JNI shared library | Kotlin |
| iOS | Static framework | Swift |
| Desktop | Shared library | Python (ctypes) |
| Embedded | Static library | C/C++ |
