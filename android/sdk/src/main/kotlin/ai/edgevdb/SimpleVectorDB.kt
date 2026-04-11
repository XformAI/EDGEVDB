package ai.edgevdb

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.io.Serializable
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.math.sqrt

/**
 * SimpleVectorDB — Pure-Kotlin in-memory vector store with cosine similarity.
 *
 * Features:
 *  - Thread-safe via ReentrantReadWriteLock
 *  - Cosine similarity search (brute-force for simplicity; HNSW for large datasets)
 *  - Optional persistence to disk via Java serialisation
 *  - Implements [EmbeddingPipeline] contract for drop-in use with [RagEngine]
 *
 * Performance characteristics:
 *  - Insert: O(1) amortised
 *  - Query:  O(n × d) where n = number of chunks, d = dimensions
 *  - Suitable for up to ~10,000 chunks on-device
 */
class SimpleVectorDB(
    override val dimensions: Int = 384,
    private val persistPath: String? = null
) : EmbeddingPipeline {

    // ── Internal storage ──────────────────────────────────────────────

    private data class Entry(
        val id:        Long,
        val text:      String,
        val embedding: FloatArray,
        val meta:      String
    ) : Serializable

    private val lock    = ReentrantReadWriteLock()
    private val entries = mutableListOf<Entry>()
    private var nextId  = 1L

    // ── EmbeddingPipeline stub ────────────────────────────────────────
    // SimpleVectorDB doesn't do text→vector itself; it delegates to an
    // external pipeline. These overrides exist so it can substitute for
    // EmbeddingPipeline in tests without an ONNX model.

    override suspend fun embed(text: String): FloatArray {
        // Hash-based deterministic fake embedding for testing
        return hashEmbedding(text)
    }

    override fun close() { /* no resources to release */ }

    // ── Insert ────────────────────────────────────────────────────────

    /**
     * Add a text chunk with its pre-computed embedding to the store.
     *
     * The [embedding] must already be L2-normalised; this method does NOT
     * normalise for you (see [l2Normalise]).
     */
    fun insert(text: String, embedding: FloatArray, meta: String = "{}"): Long {
        require(embedding.size == dimensions) {
            "Embedding dimension mismatch: expected $dimensions, got ${embedding.size}"
        }
        return lock.write {
            val id = nextId++
            entries += Entry(id, text, embedding.copyOf(), meta)
            id
        }
    }

    // ── Query ─────────────────────────────────────────────────────────

    /**
     * Return the top-[k] most similar chunks to [queryEmbedding].
     *
     * Uses cosine similarity (since all vectors should be L2-normalised,
     * this equals dot-product).
     */
    fun query(queryEmbedding: FloatArray, k: Int = 5): List<ChunkResult> {
        require(queryEmbedding.size == dimensions) {
            "Query dimension mismatch: expected $dimensions, got ${queryEmbedding.size}"
        }
        return lock.read {
            entries
                .map { entry ->
                    val score = cosineSimilarity(queryEmbedding, entry.embedding)
                    ChunkResult(
                        id    = entry.id,
                        text  = entry.text,
                        score = score,
                        meta  = entry.meta
                    )
                }
                .sortedByDescending { it.score }
                .take(k)
        }
    }

    // ── Persistence ───────────────────────────────────────────────────

    /**
     * Persist the current state to [persistPath].
     * No-op if [persistPath] was not set at construction.
     */
    suspend fun save() = withContext(Dispatchers.IO) {
        persistPath ?: return@withContext
        val file = File(persistPath)
        file.parentFile?.mkdirs()
        ObjectOutputStream(file.outputStream()).use { oos ->
            lock.read {
                oos.writeLong(nextId)
                oos.writeInt(entries.size)
                entries.forEach { oos.writeObject(it) }
            }
        }
    }

    /**
     * Restore state from [persistPath].
     * No-op if file does not exist.
     */
    suspend fun load() = withContext(Dispatchers.IO) {
        persistPath ?: return@withContext
        val file = File(persistPath)
        if (!file.exists()) return@withContext
        ObjectInputStream(file.inputStream()).use { ois ->
            lock.write {
                entries.clear()
                nextId = ois.readLong()
                val count = ois.readInt()
                repeat(count) { entries += ois.readObject() as Entry }
            }
        }
    }

    fun clear() = lock.write { entries.clear(); nextId = 1L }

    val size: Int get() = lock.read { entries.size }

    // ── Math helpers ──────────────────────────────────────────────────

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        // Both should already be L2-normalised → cosine = dot product
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot.coerceIn(-1f, 1f)
    }

    /**
     * L2-normalise [v] in-place.
     * Call this on every embedding before [insert] or [query].
     */
    fun l2Normalise(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm > 1e-9f) {
            for (i in v.indices) {
                v[i] = v[i] / norm
            }
        }
        return v
    }

    /**
     * Deterministic hash-based embedding for testing (no ONNX required).
     * Each unique string maps to a consistent 384-dim unit vector.
     */
    private fun hashEmbedding(text: String): FloatArray {
        val seed = text.hashCode().toLong()
        val v = FloatArray(dimensions) { i ->
            // LCG pseudo-random using index + seed
            val x = ((seed * 6364136223846793005L + 1442695040888963407L) xor i.toLong())
            (x and 0xFFFFF).toFloat() / 0xFFFFF.toFloat() * 2f - 1f
        }
        return l2Normalise(v)
    }
}
