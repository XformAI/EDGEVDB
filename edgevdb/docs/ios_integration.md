# iOS Integration Guide

## Prerequisites
- Xcode 14+
- iOS 15.0+ deployment target
- CMake 3.22+ (for building the xcframework)

## Building

### Option 1: Build xcframework
```bash
cd ios
chmod +x build-xcframework.sh
./build-xcframework.sh
```
This creates `ios/build/EdgeVDB.xcframework`.

### Option 2: CocoaPods / SPM (coming soon)

## Integration

### 1. Add Framework
Drag `EdgeVDB.xcframework` into your Xcode project → Frameworks.

### 2. Add Bridging Header
The framework includes `edgevdb/vectordb.h`. The Swift wrapper uses it automatically.

### 3. Add Model Files
Add `model.onnx` and `vocab.txt` to your app bundle.

### 4. Usage
```swift
import EdgeVDB

class RAGManager {
    let db: EdgeVDB
    let embedder: Embedder

    init() throws {
        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        db = try EdgeVDB(storageDir: docsURL.appendingPathComponent("edgevdb"))

        let modelURL = Bundle.main.url(forResource: "model", withExtension: "onnx")!
        let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "txt")!
        embedder = try Embedder(modelURL: modelURL, vocabURL: vocabURL)
    }

    func ingest(_ text: String, docId: UInt32) throws {
        let _ = try db.insertText(using: embedder, text: text, docId: docId, pageNumber: 0)
        try db.save()
    }

    func search(_ query: String) throws -> [ChunkResult] {
        let results = try db.queryText(using: embedder, query: query, topK: 5)
        return results.toArray()
    }
}
```
