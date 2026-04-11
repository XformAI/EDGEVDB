package ai.edgevdb

import org.junit.Assert.*
import org.junit.Test

/**
 * Basic unit tests for EdgeVDB API surface.
 * These are JVM-only tests — native JNI methods are not available
 * without an Android device. Use instrumentation tests for full coverage.
 */
class EdgeVDBTest {

    @Test
    fun `ChunkResult default values are correct`() {
        val result = ChunkResult()
        assertEquals(0L, result.id)
        assertEquals("", result.text)
        assertEquals(0f, result.score, 1e-6f)
        assertEquals("", result.meta)
        assertEquals("", result.docId)
        assertEquals(0, result.page)
    }

    @Test
    fun `DocumentChunk stores metadata`() {
        val chunk = DocumentChunk(
            text = "Hello world",
            docId = "doc1",
            chunkIndex = 3,
            page = 1,
            metadata = """{"key":"value"}"""
        )
        assertEquals("Hello world", chunk.text)
        assertEquals("doc1", chunk.docId)
        assertEquals(3, chunk.chunkIndex)
        assertEquals(1, chunk.page)
        assertEquals("""{"key":"value"}""", chunk.metadata)
    }

    @Test
    fun `QueryResult holds chunks and context`() {
        val chunks = listOf(
            ChunkResult(id = 1, text = "chunk1", score = 0.9f),
            ChunkResult(id = 2, text = "chunk2", score = 0.7f)
        )
        val result = QueryResult(
            chunks = chunks,
            contextString = "assembled context",
            latencyMs = 42
        )
        assertEquals(2, result.chunks.size)
        assertEquals("assembled context", result.contextString)
        assertEquals(42L, result.latencyMs)
    }

    @Test
    fun `ObjectRecord holds data`() {
        val record = ObjectRecord(id = 1, collection = "docs", json = """{"title":"test"}""")
        assertEquals(1L, record.id)
        assertEquals("docs", record.collection)
    }

    @Test
    fun `SyncConfig defaults`() {
        val config = SyncConfig()
        assertFalse(config.enabled)
        assertEquals("", config.deviceId)
        assertEquals("", config.syncUrl)
    }
}
