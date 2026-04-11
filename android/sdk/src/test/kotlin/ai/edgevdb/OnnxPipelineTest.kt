package ai.edgevdb

import org.junit.Assert.*
import org.junit.Test

/**
 * Tests for ONNX pipeline components that can run without Android context.
 * Full OnnxEmbeddingPipeline tests require instrumentation (Android device).
 */
class OnnxPipelineTest {

    @Test
    fun `EmbeddingException carries message and cause`() {
        val cause = RuntimeException("inner error")
        val ex = EmbeddingException("outer error", cause)
        assertEquals("outer error", ex.message)
        assertEquals(cause, ex.cause)
    }

    @Test
    fun `EmbeddingException without cause`() {
        val ex = EmbeddingException("model not found")
        assertEquals("model not found", ex.message)
        assertNull(ex.cause)
    }

    @Test
    fun `SimpleVectorDB implements EmbeddingPipeline interface`() {
        val db = SimpleVectorDB(dimensions = 64)
        assertTrue("SimpleVectorDB should implement EmbeddingPipeline",
            db is EmbeddingPipeline)
        assertEquals(64, db.dimensions)
        db.close()  // should be no-op
    }

    @Test
    fun `SimpleVectorDB embed returns correct dimensions`() = kotlinx.coroutines.test.runTest {
        val db = SimpleVectorDB(dimensions = 128)
        val embedding = db.embed("test text")
        assertEquals(128, embedding.size)

        // Verify unit length
        var norm = 0f
        for (x in embedding) norm += x * x
        assertEquals(1.0f, norm, 1e-4f)
    }
}
