package ai.edgevdb

data class SyncConfigData(
    val enabled: Boolean = false,
    val deviceId: String = "",
    val syncEndpoint: String = "",
    val syncIntervalSeconds: Int = 60,
    val syncChunks: Boolean = true,
    val syncObjects: Boolean = true
)
