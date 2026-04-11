package ai.edgevdb.demo.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Visualizes a 384-dimensional embedding vector as a radial plot.
 * Shows the first 32 dimensions for clarity.
 */
@Composable
fun EmbeddingVisualizer(embedding: FloatArray, label: String = "Embedding") {
    val primaryColor = MaterialTheme.colorScheme.primary
    Column(modifier = Modifier.padding(8.dp)) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
        ) {
            val center = Offset(size.width / 2, size.height / 2)
            val radius = size.minDimension / 2 - 20.dp.toPx()

            // Draw concentric circles
            for (i in 1..4) {
                drawCircle(
                    color = Color.White.copy(alpha = 0.1f),
                    radius = radius * i / 4,
                    center = center,
                    style = Stroke(width = 1.dp.toPx())
                )
            }

            // Draw radial lines
            val dimCount = 32  // visualize first 32 dims
            for (i in 0 until dimCount) {
                val angle = 2 * Math.PI * i / dimCount
                val x = center.x + radius * cos(angle).toFloat()
                val y = center.y + radius * sin(angle).toFloat()
                drawLine(
                    color = Color.White.copy(alpha = 0.1f),
                    start = center,
                    end = Offset(x, y),
                    strokeWidth = 1.dp.toPx()
                )
            }

            // Draw embedding polygon
            val points = mutableListOf<Offset>()
            for (i in 0 until dimCount) {
                val value = embedding.getOrNull(i) ?: 0f
                val angle = 2 * Math.PI * i / dimCount
                val r = radius * abs(value.coerceIn(-1f, 1f))
                val x = center.x + r * cos(angle).toFloat()
                val y = center.y + r * sin(angle).toFloat()
                points.add(Offset(x, y))
            }

            // Draw filled polygon
            if (points.size > 2) {
                val path = androidx.compose.ui.graphics.Path().apply {
                    moveTo(points[0].x, points[0].y)
                    for (i in 1 until points.size) {
                        lineTo(points[i].x, points[i].y)
                    }
                    close()
                }
                drawPath(
                    path = path,
                    color = primaryColor.copy(alpha = 0.3f)
                )
            }

            // Draw outline
            for (i in 0 until points.size) {
                val next = (i + 1) % points.size
                drawLine(
                    color = primaryColor,
                    start = points[i],
                    end = points[next],
                    strokeWidth = 2.dp.toPx()
                )
            }
        }
    }
}
