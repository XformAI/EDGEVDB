package ai.edgevdb

import java.util.UUID

data class EdgeVDBConfig(
    val hnswM: Int = 16,
    val efConstruction: Int = 200,
    val efSearch: Int = 64,
    val rankerAlpha: Float = 0.70f,
    val rankerBeta: Float = 0.20f,
    val rankerGamma: Float = 0.10f,
    val tokenBudget: Int = 3200,
    val embeddingThreads: Int = 2,
    val enableKnowledgeGraph: Boolean = true,
    val enableSync: Boolean = false,
    val deviceId: String = UUID.randomUUID().toString()
)
