# Android Integration Guide

## Prerequisites
- Android Studio (Arctic Fox or newer)
- NDK r25+ (tested with r27, arm64-v8a + x86_64)
- CMake 3.22+

## Setup

### Option A: Local Module
```kotlin
// settings.gradle.kts
include(":edgevdb")
project(":edgevdb").projectDir = file("path/to/edgevdb/android")

// app/build.gradle.kts
dependencies {
    implementation(project(":edgevdb"))
}
```

### Option B: Maven Central
```kotlin
dependencies {
    implementation("ai.edgevdb:edgevdb-android:1.0.0")
}
```

## Usage — Without ONNX (Pre-computed Embeddings)

Use embeddings from any provider (ML Kit, TFLite, OpenAI API, etc.):

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.ChunkResult

class MyRepository(context: Context) {
    private val db = EdgeVDB.open(context)

    fun index(text: String, embedding: FloatArray, docId: Int, page: Int): Long {
        return db.insertChunk(embedding, text, docId, page)
    }

    fun search(queryEmbedding: FloatArray, query: String): List<ChunkResult> {
        return db.queryVector(queryEmbedding, query, topK = 5).use { it.toList() }
    }

    fun storeMetadata(title: String): Long {
        return db.putObject("Document", mapOf("title" to title))
    }

    fun cleanup() {
        db.save()
        db.close()
    }
}
```

## Usage — With Built-in Embedder

Place `model.onnx` and `vocab.txt` in `app/src/main/assets/`:

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.Embedder

val embedder = Embedder.fromAssets(context)
val db = EdgeVDB.open(context)

val chunkId = db.insertText(embedder, "Neural networks classify images", docId = 1, pageNumber = 0)

db.queryText(embedder, "image classification", topK = 5).use { results ->
    results.toList().forEach { println("${it.score}: ${it.text}") }
}

embedder.destroy()
db.close()
```

## ViewModel Example

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.Embedder
import ai.edgevdb.ChunkResult

class RAGViewModel(application: Application) : AndroidViewModel(application) {
    private val db = EdgeVDB.open(application)
    private val embedder = Embedder.fromAssets(application)

    fun ingest(text: String, docId: Int) {
        // Split on word boundaries — 400 words per chunk, 50-word overlap
        val words = text.split("\\s+".toRegex())
        val chunkSize = 400
        val overlap = 50
        var page = 0
        var i = 0
        while (i < words.size) {
            val chunk = words.subList(i, minOf(i + chunkSize, words.size)).joinToString(" ")
            if (chunk.length < 20) break  // skip trivially small final fragments
            db.insertText(embedder, chunk, docId, page)
            page++
            i += chunkSize - overlap
        }
        db.save()
    }

    fun search(query: String): List<ChunkResult> {
        return db.queryText(embedder, query, topK = 5).use { it.toList() }
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

## Build Presets

```bash
# ARM64 (physical devices)
cmake --preset android-arm64 && cmake --build build/android-arm64

# x86_64 (emulators)
cmake --preset android-x86_64 && cmake --build build/android-x86_64
```

Output: `build/android-*/core/libedgevdb_shared.so` — copy to `jniLibs/`.
