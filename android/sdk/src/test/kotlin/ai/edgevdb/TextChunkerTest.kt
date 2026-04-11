package ai.edgevdb

import org.junit.Assert.*
import org.junit.Test

class TextChunkerTest {

    @Test
    fun `single chunk for short text`() {
        val chunker = TextChunker(chunkSize = 100, chunkOverlap = 10)
        val text    = "Hello world. This is a short document."
        val chunks  = chunker.chunk(text, "doc1")
        assertEquals(1, chunks.size)
        assertEquals("doc1", chunks[0].docId)
        assertEquals(0, chunks[0].chunkIndex)
    }

    @Test
    fun `multiple chunks with overlap`() {
        val chunker = TextChunker(chunkSize = 5, chunkOverlap = 2)
        val words   = (1..20).map { "word$it" }
        val text    = words.joinToString(" ")
        val chunks  = chunker.chunk(text, "doc2")

        assertTrue("Should produce multiple chunks", chunks.size > 1)
        // Each chunk should start chunkSize-overlap words after the previous
        val firstChunkWords  = chunks[0].text.split(" ")
        val secondChunkWords = chunks[1].text.split(" ")
        // Last 2 words of chunk 0 should be first 2 of chunk 1 (overlap=2)
        assertEquals(
            firstChunkWords.takeLast(2),
            secondChunkWords.take(2)
        )
    }

    @Test
    fun `overlap must be less than chunkSize`() {
        assertThrows(IllegalArgumentException::class.java) {
            TextChunker(chunkSize = 10, chunkOverlap = 10)
        }
    }
}
