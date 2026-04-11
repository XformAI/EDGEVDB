package ai.edgevdb.demo.di

import ai.edgevdb.EdgeVDB
import ai.edgevdb.OnnxEmbeddingPipeline
import ai.edgevdb.RagEngine
import ai.edgevdb.SimpleVectorDB
import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Named
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    /**
     * ONNX embedding pipeline — uses the quantized all-MiniLM-L6-v2 model
     * bundled as an Android asset.
     */
    @Provides
    @Singleton
    fun provideOnnxPipeline(@ApplicationContext ctx: Context): OnnxEmbeddingPipeline =
        OnnxEmbeddingPipeline(ctx, useNnapi = false)

    /**
     * Native EdgeVDB database stored in app's internal files directory.
     */
    @Provides
    @Singleton
    fun provideEdgeVDB(@ApplicationContext ctx: Context): EdgeVDB =
        EdgeVDB.open(
            context  = ctx,
            dbPath   = "${ctx.filesDir.absolutePath}/edgevdb",
            dims     = 384,
            enableKG = false,   // set true to enable knowledge graph
            enableOnnx = false  // embeddings done in Kotlin
        )

    /**
     * Pure-Kotlin fallback vector DB (for emulator / CI testing).
     */
    @Provides
    @Singleton
    fun provideSimpleVectorDB(@ApplicationContext ctx: Context): SimpleVectorDB =
        SimpleVectorDB(
            dimensions  = 384,
            persistPath = "${ctx.filesDir.absolutePath}/simple_vdb.dat"
        )

    /**
     * RagEngine wired to native EdgeVDB + ONNX pipeline.
     * Swap [db] for [simpleDB] to use the pure-Kotlin path.
     */
    @Provides
    @Singleton
    fun provideRagEngine(
        pipeline: OnnxEmbeddingPipeline,
        db:       EdgeVDB
    ): RagEngine = RagEngine(pipeline = pipeline, db = db)
}
