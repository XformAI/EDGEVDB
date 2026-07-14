# EdgeVDB iOS SDK

> **Production-ready iOS SDK for on-device vector database with HNSW ANN, hybrid retrieval, knowledge graph, and CRDT-based sync — all in a single zero-dependency native library.**

## Overview

The EdgeVDB iOS SDK provides a complete on-device vector database solution for iOS applications. It combines Kotlin APIs with a native C++ core via JNI, enabling:

- **HNSW ANN Index** — Approximate nearest neighbor search with logarithmic complexity
- **Hybrid Retrieval** — Multi-factor ranking combining cosine similarity, page proximity, and keyword overlap
- **Knowledge Graph** — On-device NER → entity graph → multi-hop expansion
- **Embedding Pipeline** — Optional ONNX Runtime integration (all-MiniLM-L6-v2) with CoreML acceleration
- **Object Store** — Schema-less JSON object store with typed property indexing
- **Relation Index** — Foreign key edges between objects
- **CRDT Sync** — Last-writer-wins vector clock synchronization across devices
- **Zero Dependencies** — Core library requires no external libraries (ONNX Runtime is optional)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    iOS Application                      │
├─────────────────────────────────────────────────────────┤
│                 Swift API (EdgeVDB.swift)               │
│  - EdgeVDB, Embedder, RagEngine, ChunkResult, etc.     │
├─────────────────────────────────────────────────────────┤
│               Objective-C++ Bridge (EdgeVDB.m)            │
├─────────────────────────────────────────────────────────┤
│                 EdgeVDB C++ Core                         │
│  (HNSW Index, Object Store, KG, Sync, Hybrid Ranker)    │
└─────────────────────────────────────────────────────────┘
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **HNSW ANN Index** | ✅ | 384-dim float32 vectors, M=16, ef=200/64 |
| **Hybrid Retrieval** | ✅ | α·cosine + β·page_proximity + γ·keyword |
| **Knowledge Graph** | ✅ | On-device NER → entity graph → multi-hop expansion |
| **Embedding Pipeline** | ✅ | Optional ONNX Runtime with CoreML acceleration |
| **Object Store** | ✅ | Schema-less NoSQL with typed property indexing |
| **Relation Index** | ✅ | Foreign key edges between objects |
| **CRDT Sync** | ✅ | LWW vector clock sync across devices |
| **Thread Safety** | ✅ | Actor pattern for safe concurrent access |

## Requirements

- **iOS**: 15.0+
- **Xcode**: 14.0+
- **Swift**: 5.7+
- **Platforms**: iOS, macOS, watchOS, tvOS

## Quick Start

### 1. Add Dependency

**Option A: Swift Package Manager (Recommended)**

```swift
// Package.swift or Xcode → File → Add Package Dependencies
dependencies: [
    .package(url: "https://github.com/XformAI/EDGEVDB.git", from: "0.2.0")
]
```

**Option B: CocoaPods**

```ruby
# Podfile
platform :ios, '15.0'
pod 'EdgeVDB', '~> 1.0'
```

**Option C: Build XCFramework**

```bash
cd ios
chmod +x build-xcframework.sh
./build-xcframework.sh
# Output: build/EdgeVDB.xcframework
# Drag into Xcode → Frameworks, Libraries, and Embedded Content
```

### 2. Add Model Files (Optional)

For built-in embedding, add ONNX model and vocabulary to your app bundle:

```
YourApp/
├── model.onnx                    # all-MiniLM-L6-v2 (quantized recommended)
└── vocab.txt                     # WordPiece vocabulary (30,522 tokens)
```

**Download models:**
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Export to ONNX: `python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 models/`
- Quantize to INT8: Reduces size from ~90MB to ~22MB with minimal accuracy loss

### 3. Initialize Database

```swift
import EdgeVDB

let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
let storageDir = docsDir.appendingPathComponent("edgevdb").path

let db = try EdgeVDB(
    storageDir: storageDir,
    dims: 384,
    enableKG: false
)
```

## Usage

### Without ONNX (Recommended)

Use embeddings from any provider (OpenAI, Core ML, sentence-transformers, etc.):

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
        return try db.insertChunk(text: text, embedding: embedding, docId: docId, page: page)
    }

    func search(embedding: [Float], query: String, topK: Int = 5) throws -> [ChunkResult] {
        let results = try db.queryVector(embedding: embedding, queryText: query, topK: topK)
        defer { results.close() }
        return results.toArray()
    }

    func close() {
        try? db.save()
        db.close()
    }
}
```

### With Built-in Embedder

```swift
import EdgeVDB

let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
let storageDir = docsDir.appendingPathComponent("edgevdb").path

let embedder = try Embedder(
    modelPath: Bundle.main.path(forResource: "model", ofType: "onnx")!,
    vocabPath: Bundle.main.path(forResource: "vocab", ofType: "txt")!,
    threads: 2
)

let db = try EdgeVDB(storageDir: storageDir)

let chunkId = try db.insertText(embedder: embedder, text: "Swift is powerful", docId: 1, page: 0)

let results = try db.queryText(embedder: embedder, query: "iOS development", topK: 5)
defer { results.close() }
for result in results.toArray() {
    print("[\(result.score)] \(result.text)")
}

embedder.destroy()
db.close()
```

### Object Store & Relations

```swift
import EdgeVDB

// Store object
let objId = try db.putObject(type: "Document", properties: ["title": "ML Guide", "author": "Alice"])

// Retrieve object
let obj = try db.getObject(id: objId)
print(obj)

// Add relation
try db.addRelation(name: "has_chunk", fromId: objId, toId: chunkId)

// Get targets
let targets = try db.getTargets(relation: "has_chunk", fromId: objId)
print("Targets: \(targets)")
```

### Async/Await Pattern

```swift
import EdgeVDB

class DatabaseManager {
    private let db: EdgeVDB
    private var embedder: Embedder?

    init() throws {
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let storageDir = docsDir.appendingPathComponent("edgevdb").path
        db = try EdgeVDB(storageDir: storageDir)
        
        // Optional: Initialize embedder
        if let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx"),
           let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "txt") {
            embedder = try Embedder(modelPath: modelPath, vocabPath: vocabPath, threads: 2)
        }
    }

    func insert(text: String, docId: UInt32, page: UInt32) async throws -> UInt64 {
        guard let embedder = embedder else {
            throw EdgeVDBError.embedderNotInitialized
        }
        return try db.insertText(embedder: embedder, text: text, docId: docId, page: page)
    }

    func search(query: String, topK: Int = 5) async throws -> [ChunkResult] {
        guard let embedder = embedder else {
            throw EdgeVDBError.embedderNotInitialized
        }
        let results = try db.queryText(embedder: embedder, query: query, topK: topK)
        defer { results.close() }
        return results.toArray()
    }

    deinit {
        embedder?.destroy()
        try? db.save()
        db.close()
    }
}
```

## API Reference

### EdgeVDB

Main database class.

#### Constructor

```swift
EdgeVDB(storageDir: String, dims: Int = 384, enableKG: Bool = true)
```

**Parameters:**
- `storageDir`: Directory for database files
- `dims`: Embedding dimensions (default: 384)
- `enableKG`: Enable knowledge graph (default: true)

#### Methods

##### Vector Store

**insertChunk(text:embedding:docId:page:) -> UInt64**
- Insert text with pre-computed embedding
- Returns chunk ID

```swift
let chunkId = try db.insertChunk(
    text: "Your text here",
    embedding: [0.1, 0.2, ...],  // 384-dim float array
    docId: 1,
    page: 0
)
```

**insertText(embedder:text:docId:page:) -> UInt64**
- Insert text with auto-embedding via embedder
- Returns chunk ID

```swift
let chunkId = try db.insertText(
    embedder: embedder,
    text: "Your text here",
    docId: 1,
    page: 0
)
```

**queryVector(embedding:queryText:topK:) -> QueryResults**
- Query with pre-computed embedding
- Returns QueryResults object

```swift
let results = try db.queryVector(
    embedding: [0.1, 0.2, ...],
    queryText: "search query",
    topK: 5
)
```

**queryText(embedder:query:topK:) -> QueryResults**
- Query with auto-embedding via embedder
- Returns QueryResults object

```swift
let results = try db.queryText(
    embedder: embedder,
    query: "search query",
    topK: 5
)
```

##### Object Store

**putObject(type:properties:) -> UInt64**
- Store JSON object
- Returns object ID

```swift
let docId = try db.putObject(
    type: "Document",
    properties: ["title": "My Doc", "author": "Alice"]
)
```

**getObject(id:) -> [String: Any]?**
- Retrieve object by ID
- Returns dictionary or nil if not found

```swift
if let obj = try db.getObject(id: docId) {
    print(obj["title"] as? String)
}
```

##### Relations

**addRelation(name:fromId:toId:)**
- Add typed edge between objects

```swift
try db.addRelation(name: "has_chunk", fromId: docId, toId: chunkId)
```

**getTargets(relation:fromId:) -> [UInt64]**
- Get target IDs for a relation from a source

```swift
let targets = try db.getTargets(relation: "has_chunk", fromId: docId)
```

##### Lifecycle

**save()**
- Flush all data to disk

```swift
try db.save()
```

**close()**
- Release native resources

```swift
db.close()
```

### Embedder

ONNX embedding model wrapper.

#### Constructor

```swift
Embedder(modelPath:vocabPath:threads:)
```

**Parameters:**
- `modelPath`: Path to ONNX model file
- `vocabPath`: Path to vocabulary file
- `threads`: Number of inference threads (default: 2)

#### Methods

**embed(text:) -> [Float]**
- Embed text to 384-dim vector
- Returns array of floats

```swift
let embedding = try embedder.embed("Hello world")
```

**destroy()**
- Release native resources

```swift
embedder.destroy()
```

### QueryResults

Query result container with lazy access.

#### Properties

**count** (Int): Number of results

```swift
print("Found \(results.count) results")
```

**contextString** (String): Pre-assembled RAG context

```swift
print(results.contextString)
```

#### Methods

**toArray() -> [ChunkResult]**
- Convert to array

```swift
let resultsArray = results.toArray()
```

**close()**
- Free native query handle

```swift
results.close()
```

### ChunkResult

Single query result.

#### Attributes

- **chunkId** (UInt64): Unique chunk identifier
- **text** (String): Chunk text content
- **score** (Float): Hybrid similarity score [0.0, 1.0]
- **page** (UInt32): Page number in document
- **docId** (UInt32): Document identifier

```swift
for result in results.toArray() {
    print("ID: \(result.chunkId)")
    print("Text: \(result.text)")
    print("Score: \(result.score)")
    print("Page: \(result.page)")
}
```

## Configuration

### Database Configuration

```swift
let db = try EdgeVDB(
    storageDir: storageDir,
    dims: 384,              // Embedding dimensions
    enableKG: true,        // Enable knowledge graph
    hnswM: 16,             // HNSW M parameter
    efConstruction: 200,    // HNSW ef_construction
    efSearch: 64           // HNSW ef_search
)
```

### Ranking Weights

```swift
// Customize ranking weights
let db = try EdgeVDB(
    storageDir: storageDir,
    rankerAlpha: 0.70,    // Cosine similarity
    rankerBeta: 0.20,     // Page proximity
    rankerGamma: 0.10     // Keyword overlap
)
```

## Performance Optimization

### Enable CoreML Delegate

```swift
let embedder = try Embedder(
    modelPath: modelPath,
    vocabPath: vocabPath,
    useCoreML: true  // Enable CoreML acceleration
)
```

### Use Quantized Model

- Use INT8 quantized model (~22MB) instead of FP32 (~90MB)
- Reduces memory footprint and improves inference speed
- Minimal accuracy loss (~0.1% on STS benchmarks)

### Batch Operations

```swift
// Batch insert for efficiency
let embeddings = texts.map { try embedder.embed($0) }
for (index, emb) in embeddings.enumerated() {
    try db.insertChunk(text: texts[index], embedding: emb, docId: docId, page: UInt32(index))
}
try db.save()  // Save once after all inserts
```

## Thread Safety

- EdgeVDB is not thread-safe by default
- Use a serial dispatch queue or actor for concurrent access
- Always close query results with `defer { results.close() }`
- Embedder is thread-safe for concurrent `embed()` calls

### Actor Pattern (Recommended)

```swift
import EdgeVDB

actor DatabaseActor {
    private let db: EdgeVDB

    init(storageDir: String) throws {
        db = try EdgeVDB(storageDir: storageDir)
    }

    func insert(text: String, embedding: [Float], docId: UInt32, page: UInt32) throws -> UInt64 {
        return try db.insertChunk(text: text, embedding: embedding, docId: docId, page: page)
    }

    func search(embedding: [Float], query: String, topK: Int = 5) throws -> [ChunkResult] {
        let results = try db.queryVector(embedding: embedding, queryText: query, topK: topK)
        defer { results.close() }
        return results.toArray()
    }

    func save() throws {
        try db.save()
    }

    deinit {
        db.close()
    }
}
```

## Building

### Build XCFramework

```bash
cd ios
chmod +x build-xcframework.sh
./build-xcframework.sh
```

**Output:** `build/EdgeVDB.xcframework` (universal for arm64 + x86_64 simulator)

### Manual Build

```bash
# Build for iOS (arm64)
cmake --preset ios-arm64
cmake --build build/ios-arm64

# Build for iOS Simulator (x86_64)
cmake --preset ios-simulator-x86_64
cmake --build build/ios-simulator-x86_64

# Build for macOS
cmake --preset desktop-release
cmake --build build/desktop-release
```

## Testing

### Unit Tests

```bash
# Run iOS tests
xcodebuild test -scheme EdgeVDB -destination 'platform=iOS Simulator,name=iPhone 15'

# Run macOS tests
swift test
```

### Manual Testing

```swift
import XCTest
@testable import EdgeVDB

class EdgeVDBTests: XCTestCase {
    var db: EdgeVDB!
    var tempDir: URL!

    override func setUp() async throws {
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("edgevdb_test")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        db = try EdgeVDB(storageDir: tempDir.path)
    }

    override func tearDown() async throws {
        db.close()
        try FileManager.default.removeItem(at: tempDir)
    }

    func testInsertAndQuery() throws {
        let embedding = Array(repeating: Float(0.1), count: 384)
        let chunkId = try db.insertChunk(text: "Test text", embedding: embedding, docId: 1, page: 0)
        
        let results = try db.queryVector(embedding: embedding, queryText: "test", topK: 5)
        defer { results.close() }
        
        XCTAssertGreaterThan(results.count, 0)
    }
}
```

## Troubleshooting

### Framework Not Found

**Error:** `No such module 'EdgeVDB'`

**Solutions:**
- Verify the XCFramework is added to the project
- Check "Frameworks, Libraries, and Embedded Content" in Xcode
- Ensure "Embed & Sign" is checked
- Clean build folder: Product → Clean Build Folder

### Model Loading Issues

**Error:** `Failed to load ONNX model`

**Solutions:**
- Verify model and vocab files are in the app bundle
- Check file sizes: model.onnx (~90MB or ~22MB quantized), vocab.txt (~230KB)
- Ensure files are added to "Copy Bundle Resources" in Build Phases
- Enable logging to debug

### Out of Memory Errors

**Error:** Memory warning or crash during embedding

**Solutions:**
- Reduce chunk size when ingesting large documents
- Limit concurrent embedding operations
- Use quantized ONNX model to reduce memory footprint
- Process documents in smaller batches

### Slow Inference

**Issue:** Embedding takes too long

**Solutions:**
- Use quantized INT8 model
- Enable CoreML delegate
- Reduce max sequence length
- Increase thread count (if CPU supports it)

## Best Practices

1. **Use Actors**: Wrap database access in an actor for thread safety
2. **Defer Cleanup**: Always use `defer { results.close() }` for query results
3. **Error Handling**: Properly handle and propagate errors
4. **Resource Cleanup**: Always close database and destroy embedder when done
5. **Batch Operations**: Minimize database save operations by batching inserts
6. **Quantized Models**: Use INT8 quantized models for mobile deployment
7. **Background Processing**: Use background queues for database operations

## Contributing

### Code Style

- Follow Swift API Design Guidelines
- Use SwiftLint for linting
- Add documentation comments for public APIs
- Use meaningful variable and function names

### Testing

- Write unit tests for new features
- Test on both device and simulator
- Test on multiple iOS versions if possible
- Include performance benchmarks for critical paths

## License

Apache License 2.0 — see [LICENSE](../LICENSE).

## See Also

- [../README.md](../README.md) — Project overview
- [../DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) — Build and integration guide
- [../docs/ios_integration.md](../docs/ios_integration.md) — iOS integration guide
- [../docs/architecture.md](../docs/architecture.md) — Architecture overview
- [../docs/api_reference.md](../docs/api_reference.md) — C API reference
