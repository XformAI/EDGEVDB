package ai.edgevdb.demo.ui.screen

import ai.edgevdb.demo.data.SampleDocuments
import ai.edgevdb.demo.viewmodel.IngestViewModel
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IngestScreen(
    onBack: () -> Unit,
    vm: IngestViewModel = hiltViewModel()
) {
    val state by vm.uiState.collectAsState()
    var customText by remember { mutableStateOf("") }
    var customId   by remember { mutableStateOf("custom_doc_1") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Ingest Documents") },
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
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // ── Stats card ──────────────────────────────────────────────
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Index Status", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    Text("Chunks indexed: ${state.indexedChunks}")
                    Text("Documents: ${state.ingestedDocs.size}")
                }
            }

            // ── Sample documents ────────────────────────────────────────
            Text("Sample Documents", style = MaterialTheme.typography.titleSmall)

            SampleDocuments.all.forEach { doc ->
                val ingested = doc.id in state.ingestedDocs
                Card(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(doc.title, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                "${doc.content.take(80)}…",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        if (ingested) {
                            Text("✓", color = MaterialTheme.colorScheme.tertiary)
                        }
                    }
                }
            }

            Button(
                onClick  = { vm.ingestSampleDocuments() },
                enabled  = !state.isLoading,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (state.isLoading) CircularProgressIndicator(Modifier.size(20.dp))
                else Text("Ingest All Sample Documents")
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))

            // ── Custom document ─────────────────────────────────────────
            Text("Custom Document", style = MaterialTheme.typography.titleSmall)

            OutlinedTextField(
                value         = customId,
                onValueChange = { customId = it },
                label         = { Text("Document ID") },
                modifier      = Modifier.fillMaxWidth(),
                singleLine    = true
            )

            OutlinedTextField(
                value         = customText,
                onValueChange = { customText = it },
                label         = { Text("Document Text") },
                modifier      = Modifier.fillMaxWidth().height(160.dp),
                maxLines      = 8
            )

            Button(
                onClick  = { vm.ingestCustomDocument(customId, customText) },
                enabled  = customText.isNotBlank() && !state.isLoading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Ingest Custom Document")
            }

            // ── Result / Error ──────────────────────────────────────────
            state.lastResult.takeIf { it.isNotBlank() }?.let {
                Text(it, color = MaterialTheme.colorScheme.tertiary)
            }
            state.error?.let {
                Text("Error: $it", color = MaterialTheme.colorScheme.error)
            }
        }
    }
}
