package ai.edgevdb

/**
 * Represents a stored object record in the EdgeVDB object store.
 *
 * @param id         Unique object identifier.
 * @param collection The collection/type name this object belongs to.
 * @param json       JSON-serialised properties of the object.
 */
data class ObjectRecord(
    val id:         Long,
    val collection: String,
    val json:       String
)
