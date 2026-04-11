package ai.edgevdb

/**
 * Sliding-window text chunker.
 *
 * Splits a document into overlapping word-level chunks so that context
 * at chunk boundaries is not lost. BERT models handle up to 512 tokens
 * (~350–400 English words), so [chunkSize] ≤ 300 is recommended.
 *
 * @param chunkSize    Target words per chunk.
 * @param chunkOverlap Words of overlap between consecutive chunks.
 */
class TextChunker(
    private val chunkSize:    Int = 200,
    private val chunkOverlap: Int = 40
) {
    init {
        require(chunkSize > 0)         { "chunkSize must be > 0" }
        require(chunkOverlap >= 0)     { "chunkOverlap must be >= 0" }
        require(chunkOverlap < chunkSize) { "chunkOverlap must be < chunkSize" }
    }

    /**
     * Chunk [text] into [DocumentChunk]s with sliding window.
     *
     * @param text  The full document text.
     * @param docId Identifier for the parent document.
     * @param page  Starting page number (increment externally for paged docs).
     */
    fun chunk(text: String, docId: String, page: Int = 0): List<DocumentChunk> {
        val words  = text.split(Regex("\\s+")).filter { it.isNotBlank() }
        val chunks = mutableListOf<DocumentChunk>()
        var start  = 0
        var index  = 0

        while (start < words.size) {
            val end   = minOf(start + chunkSize, words.size)
            val chunk = words.subList(start, end).joinToString(" ")

            chunks += DocumentChunk(
                text       = chunk,
                docId      = docId,
                chunkIndex = index++,
                page       = page
            )

            start += chunkSize - chunkOverlap
        }

        return chunks
    }
}
