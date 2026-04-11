package ai.edgevdb

/**
 * Configuration for the CRDT-based sync engine.
 *
 * @param enabled   Whether sync is active.
 * @param deviceId  Unique identifier for this device (used as CRDT node ID).
 * @param syncUrl   Remote sync endpoint URL (optional).
 */
data class SyncConfig(
    val enabled:  Boolean = false,
    val deviceId: String  = "",
    val syncUrl:  String  = "",
    val syncIntervalSeconds: Int = 60,
    val syncChunks: Boolean = true,
    val syncObjects: Boolean = true
)
