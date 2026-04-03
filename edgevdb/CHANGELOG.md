# Changelog

All notable changes to EdgeVDB SDK will be documented in this file.

## [1.0.0] - 2024-01-01

### Added
- Initial release of EdgeVDB SDK
- HNSW approximate nearest-neighbour index (M=16, ef_construction=200)
- Hybrid ranker with configurable cosine + page proximity + keyword weights
- Knowledge graph entity extraction and multi-hop expansion
- WordPiece tokenizer + ONNX MiniLM embedding pipeline
- Relational object store with typed property indexing
- Relation index for foreign key edges between objects
- CRDT-based data sync engine with LWW conflict resolution
- Stable C public API (vectordb.h)
- Android SDK (JNI + Kotlin)
- iOS SDK (Swift wrapper)
- Python SDK (ctypes wrapper)
- CMake unified build system with platform presets
- Binary file persistence with CRC32 integrity checks
- Full documentation and examples
