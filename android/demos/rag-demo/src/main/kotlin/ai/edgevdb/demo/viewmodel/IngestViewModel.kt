package ai.edgevdb.demo.viewmodel

import ai.edgevdb.demo.data.RagRepository
import ai.edgevdb.demo.data.SampleDocuments
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class IngestUiState(
    val isLoading:     Boolean      = false,
    val indexedChunks: Int          = 0,
    val ingestedDocs:  Set<String>  = emptySet(),
    val lastResult:    String       = "",
    val error:         String?      = null
)

@HiltViewModel
class IngestViewModel @Inject constructor(
    private val repository: RagRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(IngestUiState())
    val uiState: StateFlow<IngestUiState> = _uiState.asStateFlow()

    init {
        // Sync with repository state
        viewModelScope.launch {
            repository.indexedChunkCount.collect { count ->
                _uiState.update { it.copy(indexedChunks = count) }
            }
        }
        viewModelScope.launch {
            repository.ingestedDocIds.collect { ids ->
                _uiState.update { it.copy(ingestedDocs = ids) }
            }
        }
    }

    /** Load the built-in sample documents into the index. */
    fun ingestSampleDocuments() = viewModelScope.launch {
        _uiState.update { it.copy(isLoading = true, error = null) }
        try {
            val newChunks = repository.ingestDocuments(SampleDocuments.all)
            _uiState.update {
                it.copy(
                    isLoading  = false,
                    lastResult = "✓ Added $newChunks chunks from ${SampleDocuments.all.size} documents"
                )
            }
        } catch (e: Exception) {
            _uiState.update { it.copy(isLoading = false, error = e.message) }
        }
    }

    /** Ingest a single user-supplied document. */
    fun ingestCustomDocument(docId: String, text: String) = viewModelScope.launch {
        if (text.isBlank()) return@launch
        _uiState.update { it.copy(isLoading = true, error = null) }
        try {
            val newChunks = repository.ingestCustomDocument(docId, text)
            _uiState.update {
                it.copy(
                    isLoading  = false,
                    lastResult = "✓ Added $newChunks chunks from custom document"
                )
            }
        } catch (e: Exception) {
            _uiState.update { it.copy(isLoading = false, error = e.message) }
        }
    }
}
