package ai.edgevdb

import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class SimpleVectorDBTest {

    private lateinit var db: SimpleVectorDB

    @Before
    fun setUp() {
        db = SimpleVectorDB(dimensions = 4)  // tiny dims for testing
    }

    @Test
    fun `insert and query returns closest vector`() {
        val v1 = floatArrayOf(1f, 0f, 0f, 0f)   // normalized
        val v2 = floatArrayOf(0f, 1f, 0f, 0f)
        val v3 = floatArrayOf(0.9f, 0.1f, 0f, 0f).let { db.l2Normalise(it) }

        db.insert("Document about topic A", v1)
        db.insert("Document about topic B", v2)
        db.insert("Very similar to A",     v3)

        val results = db.query(v1, k = 2)

        assertEquals(2, results.size)
        // Top result should be the most similar to v1
        assertEquals("Document about topic A", results[0].text)
        assertTrue(results[0].score > results[1].score)
    }

    @Test
    fun `l2Normalise produces unit vector`() {
        val v = floatArrayOf(3f, 4f, 0f, 0f)
        val norm = db.l2Normalise(v)
        val magnitude = Math.sqrt((norm.map { it * it }.sum()).toDouble())
        assertEquals(1.0, magnitude, 1e-5)
    }

    @Test
    fun `empty db returns empty results`() {
        val results = db.query(floatArrayOf(1f, 0f, 0f, 0f), k = 5)
        assertTrue(results.isEmpty())
    }

    @Test
    fun `dimension mismatch throws`() {
        assertThrows(IllegalArgumentException::class.java) {
            db.insert("text", floatArrayOf(1f, 0f))  // wrong dim
        }
    }

    @Test
    fun `hashEmbedding is deterministic`() = runTest {
        val e1 = db.embed("hello world")
        val e2 = db.embed("hello world")
        assertArrayEquals(e1, e2, 1e-6f)
    }

    @Test
    fun `hashEmbedding differs for different text`() = runTest {
        val e1 = db.embed("hello world")
        val e2 = db.embed("goodbye moon")
        var diff = 0f
        for (i in e1.indices) diff += Math.abs(e1[i] - e2[i])
        assertTrue("Embeddings should differ", diff > 0.01f)
    }
}
