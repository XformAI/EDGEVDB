# iOS Integration Guide

## Prerequisites
- Xcode ≥ 14, Swift ≥ 5.7
- iOS 15.0+ deployment target
- CMake 3.22+ (for building the xcframework)

## Setup

### Option A: Swift Package Manager
```swift
// Package.swift or Xcode → File → Add Package Dependencies
dependencies: [
    .package(url: "https://github.com/edgevdb/edgevdb.git", from: "1.0.0")
]
```

### Option B: Build XCFramework
```bash
cd ios
chmod +x build-xcframework.sh
./build-xcframework.sh
# Output: build/EdgeVDB.xcframework
# Drag into Xcode → Frameworks, Libraries, and Embedded Content
```

### Option C: CocoaPods
```ruby
pod 'EdgeVDB', '~> 1.0'
```

## Usage — Without ONNX (Pre-computed Embeddings)

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

## Usage — With Built-in Embedder

Add `model.onnx` and `vocab.txt` to your app bundle:

```swift
import EdgeVDB

let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
let storageDir = docsDir.appendingPathComponent("edgevdb").path

let embedder = try Embedder(
    modelPath: Bundle.main.path(forResource: "model", ofType: "onnx")!,
    vocabPath: Bundle.main.path(forResource: "vocab", ofType: "txt")!
)
let db = try EdgeVDB(storageDir: storageDir)

let chunkId = try db.insertText(embedder: embedder, text: "Swift is powerful", docId: 1, page: 0)

let results = try db.queryText(embedder: embedder, query: "iOS development", topK: 5)
defer { results.close() }
for result in results.toArray() {
    print("\(result.score): \(result.text)")
}

embedder.destroy()
db.close()
```

## Object Store & Relations

```swift
let objId = try db.putObject(type: "Document", properties: ["title": "ML Guide"])
let obj = try db.getObject(id: objId)

try db.addRelation(name: "has_chunk", fromId: objId, toId: chunkId)
```
