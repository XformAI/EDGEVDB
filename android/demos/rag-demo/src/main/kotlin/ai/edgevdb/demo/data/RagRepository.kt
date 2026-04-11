package ai.edgevdb.demo.data

import ai.edgevdb.EdgeVDB
import ai.edgevdb.QueryResult
import ai.edgevdb.RagEngine
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Single source of truth for all RAG operations.
 *
 * Manages lifecycle of the EdgeVDB database (save on demand) and
 * exposes reactive state for the UI layer.
 */
@Singleton
class RagRepository @Inject constructor(
    private val ragEngine: RagEngine,
    private val db:        EdgeVDB
) {
    companion object {
        private const val TAG = "RagRepository"
    }

    // ── Reactive state ─────────────────────────────────────────────────

    private val _indexedChunkCount = MutableStateFlow(0)
    val indexedChunkCount: StateFlow<Int> = _indexedChunkCount

    private val _ingestedDocIds = MutableStateFlow<Set<String>>(emptySet())
    val ingestedDocIds: StateFlow<Set<String>> = _ingestedDocIds

    // ── Ingestion ──────────────────────────────────────────────────────

    /**
     * Ingest a list of sample documents. Skips already-ingested doc IDs.
     *
     * @return Total number of new chunks indexed.
     */
    suspend fun ingestDocuments(docs: List<SampleDocuments.Doc>): Int {
        var newChunks = 0
        val alreadyIngested = _ingestedDocIds.value

        docs.filter { it.id !in alreadyIngested }.forEach { doc ->
            Log.d(TAG, "Ingesting '${doc.title}'…")
            val ids = ragEngine.ingestDocument(
                text   = doc.content,
                docId  = doc.id,
                chunkSize    = 150,
                chunkOverlap = 30
            )
            newChunks += ids.size
            _ingestedDocIds.value = _ingestedDocIds.value + doc.id
        }

        _indexedChunkCount.value += newChunks
        db.save()
        return newChunks
    }

    /**
     * Ingest a single custom document provided by the user.
     */
    suspend fun ingestCustomDocument(docId: String, text: String): Int {
        val ids = ragEngine.ingestDocument(text = text, docId = docId)
        _indexedChunkCount.value += ids.size
        _ingestedDocIds.value = _ingestedDocIds.value + docId
        db.save()
        return ids.size
    }

    // ── Search ─────────────────────────────────────────────────────────

    /**
     * Run a semantic search query.
     *
     * @param query  Natural language query.
     * @param topK   Maximum results to return.
     */
    suspend fun search(query: String, topK: Int = 5): QueryResult =
        ragEngine.query(query, topK)

    // ── Stats ──────────────────────────────────────────────────────────

    suspend fun dbStats(): String = db.stats()

    /**
     * Check database on startup to restore in-memory state.
     * Call this from Application or ViewModel init.
     */
    suspend fun initialize() {
        try {
            val stats = db.stats()
            Log.d(TAG, "Database stats: $stats")
            // TODO: Parse stats to restore _indexedChunkCount and _ingestedDocIds
            // For now, we'll just log the stats
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load database stats on init", e)
        }
    }
}
