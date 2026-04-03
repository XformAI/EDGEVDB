# Android Integration Guide

## Prerequisites
- Android Studio (Arctic Fox or newer)
- NDK r25+ (arm64-v8a)
- CMake 3.22+

## Setup

### 1. Add the Library
Copy the `android/` directory into your Android project as a module, or include the AAR.

### 2. Configure Gradle
```kotlin
// settings.gradle.kts
include(":edgevdb")

// app/build.gradle.kts
dependencies {
    implementation(project(":edgevdb"))
}
```

### 3. Add Model Assets
Place ONNX model and vocab files in `app/src/main/assets/`:
- `model.onnx` — MiniLM-L6-v2 quantized
- `vocab.txt` — WordPiece vocabulary

### 4. Usage
```kotlin
class MyViewModel(application: Application) : AndroidViewModel(application) {
    private val db = EdgeVDB.open(application)
    private val embedder = Embedder.fromAssets(application)

    fun ingestDocument(text: String, docId: Int) {
        val chunks = splitIntoChunks(text, maxLen = 512)
        chunks.forEachIndexed { page, chunk ->
            db.insertText(embedder, chunk, docId, page)
        }
        db.save()
    }

    fun search(query: String): List<ChunkResult> {
        val results = db.queryText(embedder, query, topK = 5)
        return results.toList()
    }

    override fun onCleared() {
        embedder.destroy()
        db.close()
    }
}
```

## ProGuard
```
-keep class ai.edgevdb.** { *; }
```
