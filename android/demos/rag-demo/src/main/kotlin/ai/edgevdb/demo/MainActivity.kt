package ai.edgevdb.demo

import ai.edgevdb.demo.ui.screen.HomeScreen
import ai.edgevdb.demo.ui.screen.IngestScreen
import ai.edgevdb.demo.ui.screen.SearchScreen
import ai.edgevdb.demo.ui.theme.EdgeVDBTheme
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            EdgeVDBTheme {
                EdgeVDBNavGraph()
            }
        }
    }
}

@Composable
private fun EdgeVDBNavGraph() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "home") {
        composable("home") {
            HomeScreen(
                onNavigateToIngest = { navController.navigate("ingest") },
                onNavigateToSearch = { navController.navigate("search") }
            )
        }
        composable("ingest") {
            IngestScreen(onBack = { navController.popBackStack() })
        }
        composable("search") {
            SearchScreen(onBack = { navController.popBackStack() })
        }
    }
}
