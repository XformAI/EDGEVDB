package ai.edgevdb

/**
 * The complete result of a RAG query.
 *
 * @param chunks          Ranked list of matching chunks.
 * @param contextString   Chunks assembled into a single context prompt string.
 * @param queryEmbedding  The embedding vector computed for the query (useful for debugging).
 * @param latencyMs       Total retrieval latency in milliseconds.
 */
data class QueryResult(
    val chunks:         List<ChunkResult>,
    val contextString:  String,
    val queryEmbedding: FloatArray = FloatArray(0),
    val latencyMs:      Long       = 0L
)
