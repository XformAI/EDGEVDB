package ai.edgevdb.demo.ui.components

import ai.edgevdb.ChunkResult
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

@Composable
fun ChunkCard(chunk: ChunkResult) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        onClick  = { expanded = !expanded }
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically
            ) {
                // Score badge
                Surface(
                    color        = MaterialTheme.colorScheme.primaryContainer,
                    shape        = MaterialTheme.shapes.small
                ) {
                    Text(
                        text     = "${"%.3f".format(chunk.score)}",
                        style    = MaterialTheme.typography.labelMedium,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        color    = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }

                // Chunk ID
                Text(
                    "ID: ${chunk.id}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(Modifier.height(8.dp))

            Text(
                text     = chunk.text,
                style    = MaterialTheme.typography.bodySmall,
                maxLines  = if (expanded) Int.MAX_VALUE else 4,
                overflow  = TextOverflow.Ellipsis
            )

            if (chunk.docId.isNotBlank()) {
                Spacer(Modifier.height(4.dp))
                Text(
                    "Doc: ${chunk.docId}  ·  Page: ${chunk.page}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Score bar
            Spacer(Modifier.height(6.dp))
            LinearProgressIndicator(
                progress = chunk.score.coerceIn(0f, 1f),
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}
