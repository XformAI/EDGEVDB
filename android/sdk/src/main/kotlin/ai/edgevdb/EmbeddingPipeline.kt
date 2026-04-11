package ai.edgevdb

/**
 * Contract for any embedding strategy.
 *
 * Implementations must be thread-safe; [embed] may be called from multiple
 * coroutines concurrently (e.g., batch ingestion).
 */
interface EmbeddingPipeline {

    /** Dimensionality of the output embedding vectors. */
    val dimensions: Int

    /**
     * Compute a unit-length embedding for [text].
     *
     * @throws EmbeddingException if the model inference fails.
     */
    suspend fun embed(text: String): FloatArray

    /**
     * Compute embeddings for a batch of texts.
     * Default implementation calls [embed] sequentially; override for real batching.
     */
    suspend fun embedBatch(texts: List<String>): List<FloatArray> =
        texts.map { embed(it) }

    /** Release any native/GPU resources held by this pipeline. */
    fun close()
}

class EmbeddingException(message: String, cause: Throwable? = null) :
    Exception(message, cause)
