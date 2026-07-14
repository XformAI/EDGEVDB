# Changelog

All notable changes to EdgeVDB are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer.

## [0.2.0] — 2026-07-10

### Added
- **Real ONNX inference in the C++ core** — ONNX Runtime is loaded dynamically
  at startup (`dlopen`/`LoadLibrary`, no link-time dependency). With the ORT
  shared library present (default search path or `EDGEVDB_ORT_LIBRARY`) and a
  MiniLM model, `Embedder` runs true transformer inference: WordPiece →
  forward pass → attention-masked mean pooling → L2 norm. Verified by
  `tests/test_onnx_semantic.cpp`.
- **Retrieval modes** (`EvdbConfig.retrieval_mode`): `0` hybrid (union of
  HNSW vector and BM25 lexical candidates — default), `1` vector-only,
  `2` BM25-only. New in-memory **BM25 inverted index**
  (`core/src/lexical_index.hpp`); new `evdb_query_lexical()` C API and
  Python `db.query_bm25()` need **no embedding model at all**.
- **Optional page indexing** (`EvdbConfig.enable_page_index`, default on):
  when off, no page index is built or persisted and page proximity is
  excluded from ranking.
- **Opt-in retrieval algorithms** (all default-off, A/B-gated in CI, guarded
  by a runtime self-check that falls back to the original algorithm on any
  measured recall regression): diversity-aware neighbour selection
  (HNSW Alg. 4), difficulty-adaptive `ef_search`, int8 quantized traversal
  with exact float re-rank, RRF and knowledge-graph-boosted RRF ranking
  (`ranker_mode`, `rrf_k`).
- `evdb_embedder_is_semantic()`, `evdb_object_get_size()`,
  `EVDB_ERR_BUFFER_TOO_SMALL`; Python `Embedder.is_semantic`,
  `query_objects()`, working `SyncHelper`; iOS `Package.swift`;
  `bench_scale` scaling benchmark; seven new test suites
  (crc, tokenizer, persistence, concurrency, sync convergence, A/B recall,
  ONNX semantic, BM25/retrieval modes).

### Fixed
- **Data race in concurrent HNSW search** — per-query visited-list pool
  replaces shared mutable state under the reader lock.
- **Integrity** — corrupted/fake CRC-32 implementations replaced by one
  shared compile-time-generated table; CRCs now enforced on load in every
  store; loaders bounds-validate counts against file size; knowledge-graph
  loader length reads bounded (DoS fix).
- **Durability** — all saves are atomic (temp file + rename); corrupt HNSW
  files auto-rebuild from the chunk store; `evdb_open` now creates the
  storage directory (saves previously failed silently on fresh installs).
- **Sync correctness** — logical clocks stamped per chunk (not wall time),
  device-partitioned chunk IDs preserved across replicas, delete tombstones
  with resurrection suppression, idempotent edge/delta application, Lamport
  clock merge; replica convergence proven by test.
- **Tokenizer** — UTF-8-safe normalization (multi-byte sequences preserved;
  accents folded by codepoint), special-token IDs resolved from the vocab.
- Query filter is authoritative (no silent fall-through when no chunks are
  linked); relation type configurable. HNSW tombstones auto-compact at >30%.
  KG co-occurrence cleaned on chunk removal. Object/sync JSON APIs report
  `EVDB_ERR_BUFFER_TOO_SMALL` instead of silently truncating. Log macros
  honor the runtime log level. Android `settings.gradle.kts` phantom modules
  removed. MinGW DLL statically links the GCC runtime.

### Changed
- Version unified to 0.2.0 everywhere (CMake, version.h, PyPI, Gradle, docs).
- On-disk formats bumped to v2 (enforced CRC); v1 files are rejected with a
  clear message — re-ingest or rebuild (pre-1.0 policy).
- Vendored ONNX Runtime C API header replaced with the genuine v1.27 header
  (the previous stub had an incompatible `OrtApi` layout).
- Docs corrected to match implementation (sync is convergent LWW replication
  with logical clocks; no mmap claim; embedder honesty).

## [0.1.0] — initial public release
- HNSW index, hybrid ranker, knowledge graph, object/relation stores,
  LWW sync, C API, Android/iOS/Python SDKs.
