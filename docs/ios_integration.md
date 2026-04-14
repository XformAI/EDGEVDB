# iOS Integration Guide

> **Comprehensive guide for integrating EdgeVDB into iOS applications.**

## Prerequisites

- **Xcode** 14.0 or newer
- **Swift** 5.7 or newer
- **iOS** 15.0+ deployment target
- **CMake** 3.22+ (for building the xcframework)
- **Swift Package Manager** or CocoaPods

## Setup

### Option A: Swift Package Manager (Recommended)

Add EdgeVDB as a Swift Package dependency.

**Via Xcode:**
1. File → Add Package Dependencies
2. Enter URL: `https://github.com/edgevdb/edgevdb.git`
3. Select version: `1.0.6` or main branch

**Via Package.swift:**
```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/XformAI/EDGEVDB.git", from: "1.0.6")
]
```

### Option B: Build XCFramework

Build the XCFramework manually and embed it.

```bash
cd ios
chmod +x build-xcframework.sh
./build-xcframework.sh
# Output: build/EdgeVDB.xcframework
```

**Embed in Xcode:**
1. Drag `build/EdgeVDB.xcframework` into your project
2. Add to "Frameworks, Libraries, and Embedded Content"
3. Ensure "Embed & Sign" is checked

### Option C: CocoaPods

Use CocoaPods for dependency management.

```ruby
# Podfile
platform :ios, '15.0'
pod 'EdgeVDB', '~> 1.0'
```

Then run:
```bash
pod install
```

> See [../DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) §10.4 for the full podspec with C++17 and Swift subspecs.

## Model Files

### Without ONNX (Recommended)

Use embeddings from any provider (OpenAI API, Core ML, sentence-transformers, etc.). No model files needed.

### With Built-in Embedder

Add the ONNX model and vocabulary to your app bundle:

```swift
// Add to app bundle
// model.onnx
// vocab.txt
```

**Download models:**
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Export to ONNX: `python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 models/`
- Quantize to INT8: Reduces size from ~90MB to ~22MB with minimal accuracy loss

## Usage

### Without ONNX (Pre-computed Embeddings)

Use embeddings from any provider (Core ML, REST API, etc.):

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

let db = try EdgeVDB(
    storageDir: storageDir,
    dims: 384,
    enableKG: false
)

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

## Testing

### Unit Tests

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

## See Also

- [../../ios/README.md](../../ios/README.md) — iOS SDK documentation
- [../architecture.md](architecture.md) — Architecture overview
- [../api_reference.md](api_reference.md) — C API reference
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
