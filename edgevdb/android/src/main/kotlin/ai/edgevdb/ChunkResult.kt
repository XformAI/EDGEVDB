package ai.edgevdb

data class ChunkResult(
    val chunkId: Long,
    val text: String,
    val score: Float,
    val pageNumber: Int,
    val docId: Int = 0
)

class QueryResults internal constructor(private val handle: Long) : AutoCloseable {
    val count: Int get() = nativeResultCount(handle)

    fun get(index: Int): ChunkResult = ChunkResult(
        chunkId = nativeResultChunkId(handle, index),
        text = nativeResultText(handle, index),
        score = nativeResultScore(handle, index),
        pageNumber = nativeResultPage(handle, index)
    )

    val contextString: String get() = nativeResultContextString(handle)

    fun toList(): List<ChunkResult> = (0 until count).map { get(it) }

    override fun close() = nativeQueryFree(handle)

    private external fun nativeResultCount(handle: Long): Int
    private external fun nativeResultText(handle: Long, index: Int): String
    private external fun nativeResultScore(handle: Long, index: Int): Float
    private external fun nativeResultChunkId(handle: Long, index: Int): Long
    private external fun nativeResultPage(handle: Long, index: Int): Int
    private external fun nativeResultContextString(handle: Long): String
    private external fun nativeQueryFree(handle: Long)
}
