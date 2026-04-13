# Android Integration Guide

> **Comprehensive guide for integrating EdgeVDB into Android applications.**

## Prerequisites

- **Android Studio** Arctic Fox or newer
- **NDK** r25+ (tested with r27, arm64-v8a + x86_64)
- **CMake** 3.22+
- **Kotlin** 1.9.22+
- **Gradle** 8.0+
- **Min SDK** 26 (Android 8.0)

## Setup

### Option A: JitPack (Recommended)

```kotlin
// settings.gradle.kts
repositories {
    maven { url = uri("https://jitpack.io") }
}

// build.gradle.kts
dependencies {
    implementation("com.github.XformAI:EDGEVDB:v1.0.3")
}
```

No authentication required. JitPack builds directly from GitHub release tags.

### Option B: Local Module (Development)

Include the EdgeVDB Android SDK as a local Gradle module.

**1. Add to settings.gradle.kts:**
```kotlin
// settings.gradle.kts
include(":edgevdb")
project(":edgevdb").projectDir = file("path/to/edgevdb/android/sdk")
```

**2. Add to app/build.gradle.kts:**
```kotlin
// app/build.gradle.kts
dependencies {
    implementation(project(":edgevdb"))
}
```

**3. Build the native library:**
```bash
cd android
./gradlew :sdk:assembleRelease
```

## Model Files

### Without ONNX (Recommended)

Use embeddings from any provider (OpenAI API, Cohere, sentence-transformers, ML Kit, etc.). No model files needed.

### With Built-in Embedder

Place the ONNX model and vocabulary in your app's assets:

```
app/src/main/assets/
├── model.onnx                    # all-MiniLM-L6-v2 (quantized recommended)
└── vocab.txt                     # WordPiece vocabulary (30,522 tokens)
```

**Download models:**
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Export to ONNX: `python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 models/`
- Quantize to INT8: Reduces size from ~90MB to ~22MB with minimal accuracy loss

## Usage

### Without ONNX (Pre-computed Embeddings)

Use embeddings from any provider (OpenAI, Cohere, sentence-transformers, etc.):

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.ChunkResult

class MyRepository(context: Context) {
    private val db = EdgeVDB.open(context, filesDir.path + "/edgevdb")

    fun index(text: String, embedding: FloatArray, docId: Int, page: Int): Long {
        return db.insertChunk(
            chunk = DocumentChunk(text, docId.toString(), page = page),
            embedding = embedding
        )
    }

    fun search(queryEmbedding: FloatArray, query: String): List<ChunkResult> {
        return db.queryVector(queryEmbedding, topK = 5)
    }

    fun storeMetadata(title: String, author: String): Long {
        return db.putObject(
            collection = "Document",
            id = UUID.randomUUID().toString(),
            jsonBody = """{"title":"$title","author":"$author"}"""
        )
    }

    fun cleanup() {
        db.save()
        db.close()
    }
}
```

### With Built-in Embedder

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.Embedder

// Initialize embedder from assets
val embedder = Embedder.fromAssets(
    context = this,
    modelAsset = "model.onnx",
    vocabAsset = "vocab.txt",
    threads = 2
)

// Open database
val db = EdgeVDB.open(
    context = this,
    dbPath = filesDir.path + "/edgevdb",
    dims = 384,
    enableKG = false,
    enableOnnx = false
)

// Insert with auto-embedding
val chunkId = db.insertText(
    embedder,
    "Neural networks classify images",
    docId = 1,
    pageNumber = 0
)

// Query with auto-embedding
val results = db.queryText(embedder, "image classification", topK = 5)
results.forEach { result ->
    println("[${result.score}] ${result.text}")
}

// Cleanup
db.save()
embedder.destroy()
db.close()
```

### Using RAG Engine

The RAG Engine provides a complete retrieval-augmented generation pipeline:

```kotlin
import ai.edgevdb.RagEngine
import ai.edgevdb.OnnxEmbeddingPipeline

// Initialize RAG engine
val pipeline = OnnxEmbeddingPipeline(
    context = this,
    modelAssetPath = "model.onnx",
    vocabAssetPath = "vocab.txt",
    useNnapi = true  // Enable NNAPI acceleration
)

val ragEngine = RagEngine(
    pipeline = pipeline,
    db = db,
    contextMaxTokens = 2048
)

// Ingest a full document
runBlocking {
    val chunkIds = ragEngine.ingestDocument(
        text = documentText,
        docId = "doc-123",
        chunkSize = 200,
        chunkOverlap = 40
    )

    // Query
    val result = ragEngine.query(
        query = "What is the main topic?",
        topK = 5
    )

    println("Context:\n${result.contextString}")
    println("Latency: ${result.latencyMs}ms")
}
```

## ViewModel Example

```kotlin
import ai.edgevdb.EdgeVDB
import ai.edgevdb.Embedder
import ai.edgevdb.ChunkResult
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class RAGViewModel(application: Application) : AndroidViewModel(application) {
    private val db = EdgeVDB.open(
        application,
        application.filesDir.path + "/edgevdb",
        dims = 384,
        enableKG = false,
        enableOnnx = false
    )
    private val embedder = Embedder.fromAssets(application)

    fun ingest(text: String, docId: Int) = viewModelScope.launch(Dispatchers.IO) {
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

    fun search(query: String) = viewModelScope.async(Dispatchers.IO) {
        db.queryText(embedder, query, topK = 5)
    }

    override fun onCleared() {
        embedder.destroy()
        db.close()
    }
}
```

## Configuration

### Database Configuration

```kotlin
val db = EdgeVDB.open(
    context = this,
    dbPath = filesDir.path + "/edgevdb",
    dims = 384,                    // Embedding dimensions
    enableKG = true,              // Enable knowledge graph
    enableOnnx = false,           // Use Kotlin ONNX pipeline
    modelPath = null,             // Optional: path to ONNX model
    vocabPath = null              // Optional: path to vocab
)
```

### HNSW Parameters

```kotlin
val db = EdgeVDB.open(
    context = this,
    dbPath = filesDir.path + "/edgevdb",
    dims = 384,
    // HNSW parameters (via VectorDBManager config)
    // hnswM = 16, efConstruction = 200, efSearch = 64
)
```

### Ranking Weights

```kotlin
// Customize ranking weights (via VectorDBManager config)
// rankerAlpha = 0.70f  // Cosine similarity
// rankerBeta = 0.20f   // Page proximity
// rankerGamma = 0.10f  // Keyword overlap
```

## ProGuard/R8 Configuration

```
# ProGuard rules for EdgeVDB
-keep class ai.edgevdb.** { *; }
-keepclassmembers class ai.edgevdb.** { *; }
-dontwarn ai.edgevdb.**
```

## Build Configuration

### Native Library Build

```bash
cd android

# ARM64 (physical devices)
cmake --preset android-arm64
cmake --build build/android-arm64

# x86_64 (emulators)
cmake --preset android-x86_64
cmake --build build/android-x86_64
```

**Output:** `build/android-*/core/libedgevdb_shared.so`

### ABI Filters

```kotlin
// android/sdk/build.gradle.kts
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }
}
```

## Performance Optimization

### Enable NNAPI

```kotlin
val pipeline = OnnxEmbeddingPipeline(
    context = this,
    modelAssetPath = "model.onnx",
    vocabAssetPath = "vocab.txt",
    useNnapi = true  // Enable NNAPI acceleration
)
```

### Use Quantized Model

- Use INT8 quantized model (~22MB) instead of FP32 (~90MB)
- Reduces memory footprint and improves inference speed
- Minimal accuracy loss (~0.1% on STS benchmarks)

### Batch Operations

```kotlin
// Batch insert for efficiency
val embeddings = texts.map { embedder.embed(it) }
embeddings.forEachIndexed { index, emb ->
    db.insertChunk(DocumentChunk(texts[index], docId, index), emb)
}
db.save()  // Save once after all inserts
```

## Thread Safety

- All database operations are performed on `Dispatchers.IO`
- Multiple coroutines can safely perform concurrent operations
- Always call `db.close()` when done to release native resources
- Embedder is thread-safe for concurrent `embed()` calls

## Troubleshooting

### Native Library Not Found

**Error:** `java.lang.UnsatisfiedLinkError: couldn't find "edgevdb_jni"`

**Solutions:**
- Verify the native library is included in the AAR
- Check ABI filters match your device architecture
- Ensure the library is in the correct jniLibs directory

### ONNX Model Loading Issues

**Error:** `Failed to load ONNX model`

**Solutions:**
- Verify model and vocab files are in `app/src/main/assets/`
- Check file sizes: model.onnx (~90MB or ~22MB quantized), vocab.txt (~230KB)
- Enable logging: `adb logcat | grep EdgeVDB`

### Out of Memory Errors

**Error:** `OutOfMemoryError` during embedding

**Solutions:**
- Reduce chunk size when ingesting large documents
- Limit concurrent embedding operations
- Use quantized ONNX model to reduce memory footprint
- Increase heap size in manifest: `android:largeHeap="true"`

### Slow Inference

**Issue:** Embedding takes too long

**Solutions:**
- Use quantized INT8 model
- Enable NNAPI delegate
- Reduce max sequence length
- Increase thread count (if CPU supports it)

## Best Practices

1. **Use Pre-computed Embeddings**: When possible, use embeddings from your own provider to avoid ONNX overhead
2. **Batch Operations**: Minimize database save operations by batching inserts
3. **Coroutines**: Use coroutines for all database operations (they're already on Dispatchers.IO)
4. **Resource Cleanup**: Always close database and destroy embedder when done
5. **Error Handling**: Check return values and handle errors appropriately
6. **Background Threads**: Use ViewModelScope for lifecycle-aware coroutines
7. **Quantized Models**: Use INT8 quantized models for mobile deployment

## Testing

### Unit Tests

```kotlin
@Test
fun testInsertAndQuery() = runBlocking {
    val db = EdgeVDB.open(context, testDbPath)
    val embedder = Embedder.fromAssets(context)
    
    val chunkId = db.insertText(embedder, "Test text", docId = 1, pageNumber = 0)
    
    val results = db.queryText(embedder, "test", topK = 5)
    assertTrue(results.isNotEmpty())
    
    embedder.destroy()
    db.close()
}
```

### Instrumented Tests

```kotlin
@RunWith(AndroidJUnit4::class)
class EdgeVDBInstrumentedTest {
    @get:Rule
    val instantTaskExecutorRule = InstantTaskExecutorRule()

    @Test
    fun testDatabaseOperations() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = EdgeVDB.open(context, context.filesDir.path + "/test_db")
        // Test operations...
        db.close()
    }
}
```

## See Also

- [../../android/README.md](../../android/README.md) — Android SDK documentation
- [../architecture.md](architecture.md) — Architecture overview
- [../api_reference.md](api_reference.md) — C API reference
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
