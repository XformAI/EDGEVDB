package ai.edgevdb.demo

import android.app.Application
import android.util.Log
import dagger.hilt.android.HiltAndroidApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltAndroidApp
class DemoApplication : Application() {

    @Inject
    lateinit var repository: ai.edgevdb.demo.data.RagRepository

    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        // Check database stats on startup
        applicationScope.launch {
            repository.initialize()
        }
    }
}
