package ai.edgevdb

import io.mockk.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class RagEngineTest {

    private lateinit var pipeline:  EmbeddingPipeline
    private lateinit var simpleDB:  SimpleVectorDB
    private lateinit var ragEngine: RagEngine

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
        pipeline = mockk()
        simpleDB = SimpleVectorDB(dimensions = 384)
        ragEngine = RagEngine(pipeline = pipeline, simpleDB = simpleDB)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `ingestDocument chunks and embeds text`() = runTest {
        val fakeEmbedding = FloatArray(384) { 0.1f }
        coEvery { pipeline.embed(any()) } returns fakeEmbedding
        coEvery { pipeline.embedBatch(any()) } answers {
            firstArg<List<String>>().map { fakeEmbedding }
        }

        val ids = ragEngine.ingestDocument(
            text       = "Word ".repeat(200),  // 200 words
            docId      = "test_doc",
            chunkSize  = 100,
            chunkOverlap = 20
        )

        assertTrue("Should produce chunks", ids.isNotEmpty())
        assertEquals(simpleDB.size, ids.size)
        coVerify(atLeast = 1) { pipeline.embedBatch(any()) }
    }

    @Test
    fun `query returns results sorted by score`() = runTest {
        // Insert two items with known embeddings
        val embA = FloatArray(384) { if (it == 0) 1f else 0f }  // unit along dim 0
        val embB = FloatArray(384) { if (it == 1) 1f else 0f }  // unit along dim 1

        simpleDB.insert("Text about topic A", embA)
        simpleDB.insert("Text about topic B", embB)

        // Query with embedding close to A
        val queryEmb = FloatArray(384) { if (it == 0) 0.99f else if (it == 1) 0.01f else 0f }
            .let { v -> FloatArray(384).also { nv ->
                var norm = 0f; for (x in v) norm += x*x; norm = kotlin.math.sqrt(norm)
                for (i in v.indices) nv[i] = v[i] / norm
            }}

        coEvery { pipeline.embed(any()) } returns queryEmb

        val result = ragEngine.query("topic A query", topK = 2)

        assertEquals(2, result.chunks.size)
        assertEquals("Text about topic A", result.chunks[0].text)
        assertTrue(result.chunks[0].score > result.chunks[1].score)
    }
}
