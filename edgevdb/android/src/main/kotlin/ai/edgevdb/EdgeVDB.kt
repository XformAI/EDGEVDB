package ai.edgevdb

import android.content.Context
import java.io.File
import java.util.UUID

/**
 * EdgeVDB — Main Kotlin API for the on-device vector database.
 */
class EdgeVDB private constructor(private val handle: Long) {

    companion object {
        init {
            System.loadLibrary("edgevdb")
        }

        fun open(context: Context, config: EdgeVDBConfig = EdgeVDBConfig()): EdgeVDB {
            val storageDir = File(context.filesDir, "edgevdb")
            if (!storageDir.exists()) storageDir.mkdirs()

            val handle = nativeOpen(
                storageDir.absolutePath,
                config.hnswM, config.efConstruction, config.efSearch,
                config.rankerAlpha, config.rankerBeta, config.rankerGamma,
                config.tokenBudget
            )
            if (handle == 0L) throw RuntimeException("Failed to open EdgeVDB")
            return EdgeVDB(handle)
        }
    }

    fun close() = nativeClose(handle)
    fun save(): Boolean = nativeSave(handle) == 0

    // Vector store
    fun insertText(embedder: Embedder, text: String, docId: Int, pageNumber: Int): Long =
        nativeInsertText(handle, embedder.handle, text, docId, pageNumber)

    fun insertChunk(embedding: FloatArray, text: String, docId: Int, pageNumber: Int): Long =
        nativeInsertChunk(handle, embedding, text, docId, pageNumber)

    fun removeChunk(chunkId: Long): Boolean = nativeRemoveChunk(handle, chunkId) == 0

    // Query
    fun queryText(embedder: Embedder, queryText: String,
                  topK: Int = 5, useKgExpansion: Boolean = false): QueryResults {
        val qh = nativeQueryText(handle, embedder.handle, queryText, topK, useKgExpansion)
        return QueryResults(qh)
    }

    // Object store
    fun putObject(typeName: String, properties: Map<String, Any>): Long {
        val json = mapToJson(properties)
        return nativeObjectPut(handle, typeName, json)
    }

    fun getObject(id: Long): Map<String, Any>? {
        val json = nativeObjectGet(handle, id) ?: return null
        return jsonToMap(json)
    }

    fun removeObject(id: Long): Boolean = nativeObjectRemove(handle, id) == 0

    fun queryObjects(typeName: String, property: String, value: String, limit: Int = 100): List<Map<String, Any>> {
        val json = nativeObjectQuery(handle, typeName, property, value, limit) ?: return emptyList()
        return jsonArrayToList(json)
    }

    // Relations
    fun addRelation(name: String, fromId: Long, toId: Long): Boolean =
        nativeRelationAdd(handle, name, fromId, toId) == 0

    // Version
    fun version(): String = nativeVersion()

    // Private helpers
    private fun mapToJson(map: Map<String, Any>): String {
        val sb = StringBuilder("{")
        map.entries.forEachIndexed { index, (key, value) ->
            if (index > 0) sb.append(",")
            sb.append("\"$key\":")
            when (value) {
                is String -> sb.append("\"$value\"")
                is Number -> sb.append(value)
                is Boolean -> sb.append(value)
                else -> sb.append("\"$value\"")
            }
        }
        sb.append("}")
        return sb.toString()
    }

    private fun jsonToMap(json: String): Map<String, Any> {
        // Simplified JSON parsing - in production use org.json or kotlinx.serialization
        val map = mutableMapOf<String, Any>()
        // Simple key-value extraction
        val pattern = Regex("\"(\\w+)\"\\s*:\\s*(\"[^\"]*\"|[\\d.]+|true|false|null)")
        for (match in pattern.findAll(json)) {
            val key = match.groupValues[1]
            val rawValue = match.groupValues[2]
            map[key] = when {
                rawValue.startsWith("\"") -> rawValue.trim('"')
                rawValue == "true" -> true
                rawValue == "false" -> false
                rawValue == "null" -> ""
                rawValue.contains(".") -> rawValue.toDouble()
                else -> rawValue.toLongOrNull() ?: rawValue
            }
        }
        return map
    }

    private fun jsonArrayToList(json: String): List<Map<String, Any>> {
        // Simplified - split on }, { boundaries
        return json.split("},{").map { jsonToMap(it) }
    }

    // JNI declarations
    private external fun nativeOpen(storageDir: String, hnswM: Int, efConstruction: Int,
                                     efSearch: Int, alpha: Float, beta: Float,
                                     gamma: Float, tokenBudget: Int): Long
    private external fun nativeClose(handle: Long)
    private external fun nativeSave(handle: Long): Int
    private external fun nativeEmbedderCreate(modelPath: String, vocabPath: String, threads: Int): Long
    private external fun nativeEmbedderDestroy(embedderHandle: Long)
    private external fun nativeInsertText(handle: Long, embedderHandle: Long,
                                           text: String, docId: Int, pageNumber: Int): Long
    private external fun nativeInsertChunk(handle: Long, embedding: FloatArray,
                                            text: String, docId: Int, pageNumber: Int): Long
    private external fun nativeRemoveChunk(handle: Long, chunkId: Long): Int
    private external fun nativeQueryText(handle: Long, embedderHandle: Long,
                                          queryText: String, topK: Int,
                                          useKgExpansion: Boolean): Long
    private external fun nativeResultCount(queryHandle: Long): Int
    private external fun nativeResultText(queryHandle: Long, index: Int): String
    private external fun nativeResultScore(queryHandle: Long, index: Int): Float
    private external fun nativeResultPage(queryHandle: Long, index: Int): Int
    private external fun nativeResultContextString(queryHandle: Long): String
    private external fun nativeQueryFree(queryHandle: Long)
    private external fun nativeObjectPut(handle: Long, typeName: String, jsonProperties: String): Long
    private external fun nativeObjectGet(handle: Long, id: Long): String?
    private external fun nativeObjectRemove(handle: Long, id: Long): Int
    private external fun nativeObjectQuery(handle: Long, typeName: String,
                                            property: String, value: String, limit: Int): String?
    private external fun nativeRelationAdd(handle: Long, name: String, fromId: Long, toId: Long): Int
    private external fun nativeVersion(): String
}
