package ai.edgevdb

data class ObjectRecord(
    val id: Long = 0,
    val typeName: String = "",
    val properties: Map<String, Any> = emptyMap()
)

data class SyncConfig(
    val enabled: Boolean = false,
    val deviceId: String = "",
    val syncEndpoint: String = "",
    val syncIntervalSeconds: Int = 60,
    val syncChunks: Boolean = true,
    val syncObjects: Boolean = true
)

data class SyncResult(val applied: Int, val skipped: Int)
