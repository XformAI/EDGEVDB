package ai.edgevdb

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json

/**
 * Main entry point for the EdgeVDB Android SDK.
 *
 * Wraps the native C++ EdgeVDB library via JNI.
 * All database operations are performed on [Dispatchers.IO].
 *
 * Usage:
 * ```kotlin
 * val db = EdgeVDB.open(context, dbPath = filesDir.path + "/mydb")
 * val chunkId = db.insertChunk(chunk, embedding)
 * val results = db.queryVector(embedding, topK = 5)
 * db.save()
 * db.close()
 * ```
 */
class EdgeVDB private constructor(private var nativeHandle: Long) {

    companion object {
        private const val TAG = "EdgeVDB"

        init {
            System.loadLibrary("edgevdb_jni")
        }

        /**
         * Open (or create) an EdgeVDB database.
         *
         * @param context      Android context (used to locate asset files).
         * @param dbPath       Writable directory path for database files.
         * @param dims         Embedding dimensions (default: 384 for all-MiniLM-L6-v2).
         * @param enableKG     Enable on-device knowledge graph (NER + entity graph).
         * @param enableOnnx   Pass `true` only when using the C++ embedder; for Kotlin
         *                     ONNX pipeline leave this `false`.
         * @param modelPath    Optional path to ONNX model (when enableOnnx = true).
         * @param vocabPath    Optional path to vocab.txt (when enableOnnx = true).
         */
        fun open(
            context: Context,
            dbPath: String,
            dims: Int = 384,
            enableKG: Boolean = false,
            enableOnnx: Boolean = false,
            modelPath: String? = null,
            vocabPath: String? = null
        ): EdgeVDB {
            val handle = nativeOpen(
                dbPath, dims, enableKG, enableOnnx,
                modelPath ?: "", vocabPath ?: ""
            )
            require(handle != 0L) { "Failed to open EdgeVDB at $dbPath" }
            return EdgeVDB(handle)
        }

        // ── JNI declarations ──────────────────────────────────────────
        @JvmStatic private external fun nativeOpen(
            path: String, dims: Int,
            enableKG: Boolean, enableOnnx: Boolean,
            modelPath: String, vocabPath: String
        ): Long

        @JvmStatic private external fun nativeClose(handle: Long)
        @JvmStatic private external fun nativeSave(handle: Long): Int
        @JvmStatic private external fun nativeInsertText(handle: Long, text: String, meta: String): Long
        @JvmStatic private external fun nativeInsertChunk(handle: Long, text: String, embedding: FloatArray, meta: String): Long
        @JvmStatic private external fun nativeQueryText(handle: Long, query: String, topK: Int): String
        @JvmStatic private external fun nativeQueryVector(handle: Long, embedding: FloatArray, topK: Int): String
        @JvmStatic private external fun nativeObjectPut(handle: Long, collection: String, id: String, json: String): Int
        @JvmStatic private external fun nativeObjectGet(handle: Long, collection: String, id: String): String
        @JvmStatic private external fun nativeRelationAdd(handle: Long, fromCol: String, fromId: String, relType: String, toCol: String, toId: String): Int
        @JvmStatic private external fun nativeGetStats(handle: Long): String
    }

    private val json = Json { ignoreUnknownKeys = true }
    @Volatile private var closed = false

    private fun requireOpen() {
        check(!closed) { "EdgeVDB instance is already closed" }
    }

    // ── Vector Store ──────────────────────────────────────────────────

    /**
     * Insert a text chunk with a pre-computed embedding.
     *
     * @return The assigned chunk ID (positive long).
     */
    suspend fun insertChunk(chunk: DocumentChunk, embedding: FloatArray): Long =
        withContext(Dispatchers.IO) {
            requireOpen()
            Log.d(TAG, "EdgeVDB.insertChunk: text='${chunk.text.take(30)}…', embedding size=${embedding.size}, L2 norm=${embedding.map { it * it }.sum().let { kotlin.math.sqrt(it) }}")
            val id = nativeInsertChunk(nativeHandle, chunk.text, embedding, chunk.metadata)
            Log.d(TAG, "EdgeVDB.insertChunk: returned id=$id")
            id
        }

    /**
     * Insert text using the C++-side ONNX embedder (when [enableOnnx] was true at open).
     */
    suspend fun insertText(text: String, meta: String = "{}"): Long =
        withContext(Dispatchers.IO) {
            requireOpen()
            nativeInsertText(nativeHandle, text, meta)
        }

    /**
     * Search the index with a pre-computed query embedding.
     *
     * @param embedding  384-dim L2-normalised query vector.
     * @param topK       Maximum number of results to return.
     */
    suspend fun queryVector(embedding: FloatArray, topK: Int = 5): List<ChunkResult> =
        withContext(Dispatchers.IO) {
            requireOpen()
            Log.d(TAG, "EdgeVDB.queryVector: embedding size=${embedding.size}, L2 norm=${embedding.map { it * it }.sum().let { kotlin.math.sqrt(it) }}, topK=$topK")
            val jsonStr = nativeQueryVector(nativeHandle, embedding, topK)
            Log.d(TAG, "EdgeVDB.queryVector: native result=$jsonStr")
            val results = parseResults(jsonStr)
            Log.d(TAG, "EdgeVDB.queryVector: parsed ${results.size} results")
            results
        }

    /**
     * Search using the C++-side embedder (when [enableOnnx] was true at open).
     */
    suspend fun queryText(query: String, topK: Int = 5): List<ChunkResult> =
        withContext(Dispatchers.IO) {
            requireOpen()
            val jsonStr = nativeQueryText(nativeHandle, query, topK)
            parseResults(jsonStr)
        }

    // ── Object Store ──────────────────────────────────────────────────

    /** Store an arbitrary JSON-serialisable object. */
    suspend fun putObject(collection: String, id: String, jsonBody: String): Boolean =
        withContext(Dispatchers.IO) {
            requireOpen()
            nativeObjectPut(nativeHandle, collection, id, jsonBody) == 0
        }

    /** Retrieve a stored object as a JSON string. */
    suspend fun getObject(collection: String, id: String): String? =
        withContext(Dispatchers.IO) {
            requireOpen()
            val result = nativeObjectGet(nativeHandle, collection, id)
            if (result == "{}") null else result
        }

    // ── Relations ─────────────────────────────────────────────────────

    /** Add a typed edge between two stored objects. */
    suspend fun addRelation(
        fromCollection: String, fromId: String,
        relationType: String,
        toCollection: String, toId: String
    ): Boolean = withContext(Dispatchers.IO) {
        requireOpen()
        nativeRelationAdd(nativeHandle,
            fromCollection, fromId, relationType,
            toCollection, toId) == 0
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    /** Flush all in-memory state to disk. */
    suspend fun save() = withContext(Dispatchers.IO) {
        requireOpen()
        nativeSave(nativeHandle)
    }

    /** Close the database and release native resources. */
    fun close() {
        if (!closed) {
            closed = true
            nativeClose(nativeHandle)
            nativeHandle = 0L
        }
    }

    /** Return database statistics as a JSON string. */
    suspend fun stats(): String = withContext(Dispatchers.IO) {
        requireOpen()
        nativeGetStats(nativeHandle)
    }

    // ── Serialisation helper ──────────────────────────────────────────

    private fun parseResults(jsonStr: String): List<ChunkResult> = try {
        Log.d(TAG, "Parsing JSON: $jsonStr")
        val result = json.decodeFromString<List<ChunkResult>>(jsonStr)
        Log.d(TAG, "Parsed ${result.size} ChunkResult objects")
        result
    } catch (e: Exception) {
        Log.e(TAG, "Failed to parse JSON: $jsonStr", e)
        emptyList()
    }
}
