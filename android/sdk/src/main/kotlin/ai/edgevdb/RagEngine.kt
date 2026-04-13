package ai.edgevdb

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * RagEngine — Orchestrates the complete RAG pipeline.
 *
 * Combines an [EmbeddingPipeline] with an [EdgeVDB] (or [SimpleVectorDB])
 * backend to provide a one-stop API for:
 *  - Document ingestion (chunk → embed → index)
 *  - Semantic query (embed query → HNSW search → hybrid re-rank → context)
 *
 * @param pipeline      Embedding strategy (ONNX or SimpleVectorDB).
 * @param db            EdgeVDB native backend (nullable for pure-Kotlin mode).
 * @param simpleDB      Pure-Kotlin fallback (used when [db] is null).
 * @param contextMaxTokens  Maximum tokens in the assembled RAG context string.
 */
class RagEngine(
    private val pipeline:         EmbeddingPipeline,
    private val db:               EdgeVDB? = null,
    private val simpleDB:         SimpleVectorDB? = null,
    private val contextMaxTokens: Int = 2048
) {
    companion object {
        private const val TAG = "RagEngine"
        private const val WORDS_PER_TOKEN_APPROX = 1.3f  // rough approximation
    }

    init {
        require(db != null || simpleDB != null) {
            "Provide either an EdgeVDB instance (native) or a SimpleVectorDB instance."
        }
    }

    // ── Ingestion ─────────────────────────────────────────────────────

    /**
     * Ingest a single [DocumentChunk]: embed its text and add it to the index.
     *
     * @param precomputedEmbedding Optional pre-computed embedding to avoid redundant computation.
     * @return The assigned chunk ID.
     */
    suspend fun ingestChunk(chunk: DocumentChunk, precomputedEmbedding: FloatArray? = null): Long {
        val embedding = precomputedEmbedding ?: pipeline.embed(chunk.text)
        Log.d(TAG, "Ingesting chunk: docId=${chunk.docId}, chunkIndex=${chunk.chunkIndex}, embedding size=${embedding.size}")
        return if (db != null) {
            val id = db.insertChunk(chunk, embedding)
            Log.d(TAG, "Inserted chunk into EdgeVDB: id=$id")
            id
        } else {
            simpleDB!!.insert(chunk.text, embedding, chunk.metadata).also {
                Log.d(TAG, "SimpleVectorDB insert id=$it")
            }
        }
    }

    /**
     * Ingest a full document by chunking it via [TextChunker] and embedding each chunk.
     *
     * @param text          Raw document text.
     * @param docId         Unique document identifier.
     * @param chunkSize     Target words per chunk.
     * @param chunkOverlap  Word overlap between adjacent chunks.
     * @return List of assigned chunk IDs.
     */
    suspend fun ingestDocument(
        text: String,
        docId: String,
        chunkSize: Int = 200,
        chunkOverlap: Int = 40
    ): List<Long> = withContext(Dispatchers.Default) {
        val chunker = TextChunker(chunkSize, chunkOverlap)
        val chunks  = chunker.chunk(text, docId)
        Log.d(TAG, "Ingesting document '$docId': ${chunks.size} chunks")

        // Batch embed for efficiency
        val embeddings = pipeline.embedBatch(chunks.map { it.text })

        chunks.zip(embeddings).map { (chunk, emb) ->
            ingestChunk(chunk.copy(
                // Store embedding stats in metadata for debugging
                metadata = """{"docId":"${chunk.docId}","chunkIndex":${chunk.chunkIndex},"page":${chunk.page}}"""
            ), emb)
        }
    }

    // ── Query ─────────────────────────────────────────────────────────

    /**
     * Perform a full RAG query: embed the query → search → assemble context.
     *
     * @param query  Natural language query string.
     * @param topK   Number of top chunks to retrieve.
     * @return [QueryResult] containing ranked chunks and assembled context.
     */
    suspend fun query(query: String, topK: Int = 5): QueryResult {
        val t0        = System.currentTimeMillis()
        Log.d(TAG, "Querying: '${query.take(40)}…'")
        val embedding = pipeline.embed(query)
        Log.d(TAG, "Query embedding size: ${embedding.size}, L2 norm: ${embedding.map { it * it }.sum().let { kotlin.math.sqrt(it) }}")

        val chunks: List<ChunkResult> = if (db != null) {
            Log.d(TAG, "Querying EdgeVDB with topK=$topK")
            db.queryVector(embedding, topK, query)
        } else {
            Log.d(TAG, "Querying SimpleVectorDB with topK=$topK")
            simpleDB!!.query(embedding, topK)
        }

        Log.d(TAG, "Query returned ${chunks.size} chunks")
        chunks.forEachIndexed { i, chunk ->
            Log.d(TAG, "  Result $i: score=${chunk.score}, text='${chunk.text.take(30)}…'")
        }

        val context    = assembleContext(chunks)
        val latencyMs  = System.currentTimeMillis() - t0

        Log.d(TAG, "Query '${query.take(40)}…' returned ${chunks.size} chunks in ${latencyMs}ms")
        return QueryResult(
            chunks         = chunks,
            contextString  = context,
            queryEmbedding = embedding,
            latencyMs      = latencyMs
        )
    }

    // ── Context assembly ──────────────────────────────────────────────

    private fun assembleContext(chunks: List<ChunkResult>): String {
        val sb         = StringBuilder()
        var tokenCount = 0
        val maxWords   = (contextMaxTokens / WORDS_PER_TOKEN_APPROX).toInt()

        chunks.forEachIndexed { idx, chunk ->
            val chunkWords = chunk.text.split(Regex("\\s+")).size
            if (tokenCount + chunkWords > maxWords) return@forEachIndexed
            sb.appendLine("[${idx + 1}] (score: ${"%.3f".format(chunk.score)}) ${chunk.text}")
            sb.appendLine()
            tokenCount += chunkWords
        }

        return sb.toString().trim()
    }
}
