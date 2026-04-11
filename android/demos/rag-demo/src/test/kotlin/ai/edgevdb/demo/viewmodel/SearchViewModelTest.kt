package ai.edgevdb.demo.viewmodel

import ai.edgevdb.ChunkResult
import ai.edgevdb.QueryResult
import ai.edgevdb.demo.data.RagRepository
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
class SearchViewModelTest {

    private lateinit var viewModel: SearchViewModel
    private lateinit var repository: RagRepository

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
        repository = mockk()
        viewModel = SearchViewModel(repository)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial state is empty`() {
        val state = viewModel.uiState.value
        assertEquals("", state.query)
        assertFalse(state.isSearching)
        assertFalse(state.hasSearched)
        assertTrue(state.results.isEmpty())
    }

    @Test
    fun `onQueryChanged updates query`() {
        viewModel.onQueryChanged("test query")
        assertEquals("test query", viewModel.uiState.value.query)
    }

    @Test
    fun `search with blank query does nothing`() = runTest {
        viewModel.onQueryChanged("   ")
        viewModel.search()
        advanceUntilIdle()

        coVerify(exactly = 0) { repository.search(any(), any()) }
    }

    @Test
    fun `search calls repository and updates state`() = runTest {
        val mockResults = listOf(
            ChunkResult(id = 1, text = "result1", score = 0.9f),
            ChunkResult(id = 2, text = "result2", score = 0.8f)
        )
        val mockQueryResult = QueryResult(
            chunks = mockResults,
            contextString = "context",
            latencyMs = 50
        )
        coEvery { repository.search("test", 5) } returns mockQueryResult

        viewModel.onQueryChanged("test")
        viewModel.search()
        advanceUntilIdle()

        coVerify { repository.search("test", 5) }
        val state = viewModel.uiState.value
        assertTrue(state.hasSearched)
        assertEquals(2, state.results.size)
        assertEquals(50L, state.latencyMs)
    }

    @Test
    fun `clearResults resets state`() {
        viewModel.clearResults()
        val state = viewModel.uiState.value
        assertFalse(state.hasSearched)
        assertTrue(state.results.isEmpty())
        assertEquals("", state.context)
    }
}
