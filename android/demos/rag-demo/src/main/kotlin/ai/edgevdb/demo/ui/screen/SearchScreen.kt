package ai.edgevdb.demo.ui.screen

import ai.edgevdb.demo.viewmodel.SearchViewModel
import ai.edgevdb.demo.ui.components.ChunkCard
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchScreen(
    onBack: () -> Unit,
    vm: SearchViewModel = hiltViewModel()
) {
    val state by vm.uiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Semantic Search") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            // ── Search bar ──────────────────────────────────────────────
            Row(
                modifier              = Modifier.fillMaxWidth(),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedTextField(
                    value         = state.query,
                    onValueChange = { vm.onQueryChanged(it) },
                    label         = { Text("Ask anything…") },
                    modifier      = Modifier.weight(1f),
                    singleLine    = true,
                    trailingIcon  = {
                        if (state.isSearching)
                            CircularProgressIndicator(modifier = Modifier.size(24.dp))
                    }
                )
                IconButton(
                    onClick  = { vm.search() },
                    enabled  = state.query.isNotBlank() && !state.isSearching
                ) {
                    Icon(Icons.Default.Search, "Search")
                }
            }

            // ── Pre-defined example queries ─────────────────────────────
            Row(
                modifier              = Modifier.padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                listOf("What is RAG?", "HNSW algorithm", "Kotlin coroutines").forEach { q ->
                    AssistChip(
                        onClick = {
                            vm.onQueryChanged(q)
                            vm.search()
                        },
                        label = { Text(q, style = MaterialTheme.typography.labelSmall) }
                    )
                }
            }

            // ── Results ─────────────────────────────────────────────────
            if (state.hasSearched) {
                state.latencyMs.takeIf { it > 0 }?.let {
                    Text(
                        "Found ${state.results.size} results in ${it}ms",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 4.dp)
                    )
                }

                if (state.results.isEmpty()) {
                    Box(Modifier.fillMaxWidth().padding(32.dp), Alignment.Center) {
                        Text("No results — try ingesting documents first.")
                    }
                } else {
                    LazyColumn(
                        modifier              = Modifier.fillMaxSize(),
                        verticalArrangement   = Arrangement.spacedBy(8.dp),
                        contentPadding        = PaddingValues(vertical = 8.dp)
                    ) {
                        items(state.results) { chunk ->
                            ChunkCard(chunk = chunk)
                        }
                    }
                }
            }

            state.error?.let {
                Text("Error: $it", color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(top = 8.dp))
            }
        }
    }
}
