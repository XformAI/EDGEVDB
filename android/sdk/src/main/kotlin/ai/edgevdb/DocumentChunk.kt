package ai.edgevdb

/**
 * A chunk of text ready for embedding and indexing.
 *
 * @param text       The chunk text (≤ 512 tokens recommended).
 * @param docId      Identifier of the parent document.
 * @param chunkIndex Position of this chunk within its document.
 * @param page       Page number (0-indexed; 0 for non-paged content).
 * @param metadata   Optional extra metadata as a JSON string.
 */
data class DocumentChunk(
    val text:       String,
    val docId:      String,
    val chunkIndex: Int    = 0,
    val page:       Int    = 0,
    val metadata:   String = "{}"
)
