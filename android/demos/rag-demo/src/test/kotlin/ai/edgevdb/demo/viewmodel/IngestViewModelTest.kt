package ai.edgevdb.demo.viewmodel

import ai.edgevdb.demo.data.RagRepository
import io.mockk.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class IngestViewModelTest {

    private lateinit var viewModel: IngestViewModel
    private lateinit var repository: RagRepository

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
        repository = mockk()
        viewModel = IngestViewModel(repository)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial state has zero chunks`() = runTest {
        val state = viewModel.uiState.value
        assertEquals(0, state.indexedChunks)
        assertFalse(state.isLoading)
    }

    @Test
    fun `ingestSampleDocuments calls repository`() = runTest {
        coEvery { repository.ingestDocuments(any()) } returns 5
        coEvery { repository.indexedChunkCount } returns flowOf(5)
        coEvery { repository.ingestedDocIds } returns flowOf(setOf("doc1"))

        viewModel.ingestSampleDocuments()
        advanceUntilIdle()

        coVerify { repository.ingestDocuments(any()) }
        assertFalse(viewModel.uiState.value.isLoading)
    }

    @Test
    fun `ingestCustomDocument with blank text does nothing`() = runTest {
        viewModel.ingestCustomDocument("id", "")
        advanceUntilIdle()

        coVerify(exactly = 0) { repository.ingestCustomDocument(any(), any()) }
    }

    @Test
    fun `ingestCustomDocument calls repository with valid text`() = runTest {
        coEvery { repository.ingestCustomDocument(any(), any()) } returns 3
        coEvery { repository.indexedChunkCount } returns flowOf(3)

        viewModel.ingestCustomDocument("custom_doc", "Some text content")
        advanceUntilIdle()

        coVerify { repository.ingestCustomDocument("custom_doc", "Some text content") }
    }
}
