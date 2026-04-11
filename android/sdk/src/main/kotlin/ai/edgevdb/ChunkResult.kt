package ai.edgevdb

import kotlinx.serialization.Serializable

/**
 * A single result returned from a vector search query.
 *
 * @param id      Unique chunk identifier in the database.
 * @param text    The original text content of this chunk.
 * @param score   Hybrid similarity score [0.0, 1.0]; higher is more relevant.
 * @param meta    Arbitrary JSON metadata string attached at insertion.
 * @param docId   Document identifier this chunk belongs to.
 * @param page    Page number within the document (0-indexed).
 */
@Serializable
data class ChunkResult(
    val id:    Long   = 0L,
    val text:  String = "",
    val score: Float  = 0f,
    val meta:  String = "",
    val docId: String = "",
    val page:  Int    = 0
)
