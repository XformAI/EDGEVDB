package ai.edgevdb.demo.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary   = Color(0xFF6DD5FA),
    secondary = Color(0xFF2980B9),
    tertiary  = Color(0xFF27AE60),
    surface   = Color(0xFF1A1A2E),
    background = Color(0xFF0F0F1A)
)

@Composable
fun EdgeVDBTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content     = content
    )
}
