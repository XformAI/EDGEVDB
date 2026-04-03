package ai.edgevdb

import org.junit.Test
import org.junit.Assert.*

/**
 * Android unit tests for EdgeVDB SDK.
 * Note: These run on JVM, not on device. Full integration tests require androidTest/.
 */
class EdgeVDBTest {

    @Test
    fun `config has correct defaults`() {
        val config = EdgeVDBConfig()
        assertEquals(16, config.hnswM)
        assertEquals(200, config.efConstruction)
        assertEquals(64, config.efSearch)
        assertEquals(0.70f, config.rankerAlpha, 0.001f)
        assertEquals(0.20f, config.rankerBeta, 0.001f)
        assertEquals(0.10f, config.rankerGamma, 0.001f)
        assertEquals(3200, config.tokenBudget)
        assertTrue(config.enableKnowledgeGraph)
        assertFalse(config.enableSync)
    }

    @Test
    fun `ObjectRecord data class works`() {
        val record = ObjectRecord(
            id = 1,
            typeName = "Document",
            properties = mapOf("title" to "Test", "author" to "Alice")
        )
        assertEquals(1L, record.id)
        assertEquals("Document", record.typeName)
        assertEquals("Test", record.properties["title"])
    }

    @Test
    fun `ChunkResult data class works`() {
        val result = ChunkResult(
            chunkId = 42,
            text = "Hello world",
            score = 0.95f,
            pageNumber = 3,
            docId = 1
        )
        assertEquals(42L, result.chunkId)
        assertEquals("Hello world", result.text)
        assertEquals(0.95f, result.score, 0.001f)
    }

    @Test
    fun `SyncConfig defaults`() {
        val config = SyncConfigData()
        assertFalse(config.enabled)
        assertEquals(60, config.syncIntervalSeconds)
        assertTrue(config.syncChunks)
        assertTrue(config.syncObjects)
    }
}
