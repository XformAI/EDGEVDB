package ai.edgevdb.demo.data

import org.junit.Assert.*
import org.junit.Test

class SampleDocumentsTest {

    @Test
    fun `sample documents are not empty`() {
        assertTrue(SampleDocuments.all.isNotEmpty())
    }

    @Test
    fun `each document has valid id and content`() {
        SampleDocuments.all.forEach { doc ->
            assertFalse(doc.id.isBlank())
            assertFalse(doc.title.isBlank())
            assertFalse(doc.content.isBlank())
        }
    }

    @Test
    fun `document ids are unique`() {
        val ids = SampleDocuments.all.map { it.id }
        assertEquals(ids.size, ids.toSet().size)
    }

    @Test
    fun `document content is substantial`() {
        SampleDocuments.all.forEach { doc ->
            assertTrue(doc.content.length > 100)
        }
    }
}
