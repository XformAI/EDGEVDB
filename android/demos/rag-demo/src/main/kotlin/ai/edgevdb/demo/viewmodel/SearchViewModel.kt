package ai.edgevdb.demo.viewmodel

import ai.edgevdb.ChunkResult
import ai.edgevdb.QueryResult
import ai.edgevdb.demo.data.RagRepository
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SearchUiState(
    val query:       String           = "",
    val isSearching: Boolean          = false,
    val results:     List<ChunkResult> = emptyList(),
    val context:     String           = "",
    val latencyMs:   Long             = 0L,
    val error:       String?          = null,
    val hasSearched: Boolean          = false
)

@HiltViewModel
class SearchViewModel @Inject constructor(
    private val repository: RagRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(SearchUiState())
    val uiState: StateFlow<SearchUiState> = _uiState.asStateFlow()

    fun onQueryChanged(q: String) {
        _uiState.update { it.copy(query = q) }
    }

    fun search(topK: Int = 5) = viewModelScope.launch {
        val query = _uiState.value.query.trim()
        if (query.isBlank()) return@launch

        _uiState.update { it.copy(isSearching = true, error = null) }
        try {
            val result: QueryResult = repository.search(query, topK)
            _uiState.update {
                it.copy(
                    isSearching = false,
                    results     = result.chunks,
                    context     = result.contextString,
                    latencyMs   = result.latencyMs,
                    hasSearched = true
                )
            }
        } catch (e: Exception) {
            _uiState.update { it.copy(isSearching = false, error = e.message) }
        }
    }

    fun clearResults() {
        _uiState.update { it.copy(results = emptyList(), context = "", hasSearched = false) }
    }
}
