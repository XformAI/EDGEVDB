# EdgeVDB Android SDK

Android library providing on-device vector search, object storage, and sync capabilities.

## Setup

1. Add the EdgeVDB AAR to your project
2. Place `model.onnx` and `vocab.txt` in `app/src/main/assets/`
3. Initialize in your Application or ViewModel

## Usage

```kotlin
val db = EdgeVDB.open(context)
val embedder = Embedder.fromAssets(context)

// Insert document chunks
db.insertText(embedder, "Your text here...", docId = 1, pageNumber = 0)

// Query
val results = db.queryText(embedder, "search query", topK = 5)
val context = results.contextString

// Object store
val id = db.putObject("Document", mapOf("title" to "Test", "author" to "Alice"))

db.save()
db.close()
```
