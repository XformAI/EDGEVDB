package ai.edgevdb

import android.content.Context
import java.io.File
import java.io.FileOutputStream

/**
 * ONNX embedding wrapper for Android.
 */
class Embedder private constructor(internal val handle: Long) {

    companion object {
        fun fromAssets(context: Context,
                       modelAsset: String = "model.onnx",
                       vocabAsset: String = "vocab.txt",
                       threads: Int = 2): Embedder {
            val filesDir = File(context.filesDir, "edgevdb_models")
            if (!filesDir.exists()) filesDir.mkdirs()

            val modelFile = File(filesDir, modelAsset)
            val vocabFile = File(filesDir, vocabAsset)

            // Copy from assets if not already extracted
            if (!modelFile.exists()) {
                context.assets.open(modelAsset).use { input ->
                    FileOutputStream(modelFile).use { output -> input.copyTo(output) }
                }
            }
            if (!vocabFile.exists()) {
                context.assets.open(vocabAsset).use { input ->
                    FileOutputStream(vocabFile).use { output -> input.copyTo(output) }
                }
            }

            return fromFiles(modelFile.absolutePath, vocabFile.absolutePath, threads)
        }

        fun fromFiles(modelPath: String, vocabPath: String, threads: Int = 2): Embedder {
            System.loadLibrary("edgevdb")
            val handle = nativeEmbedderCreate(modelPath, vocabPath, threads)
            if (handle == 0L) throw RuntimeException("Failed to create embedder")
            return Embedder(handle)
        }

        @JvmStatic
        private external fun nativeEmbedderCreate(modelPath: String, vocabPath: String, threads: Int): Long
    }

    fun embed(text: String): FloatArray {
        val result = FloatArray(384)
        val err = nativeEmbedText(handle, text, result)
        if (err != 0) throw RuntimeException("Embedding failed with error $err")
        return result
    }

    fun destroy() {
        nativeEmbedderDestroy(handle)
    }

    private external fun nativeEmbedText(handle: Long, text: String, out: FloatArray): Int
    private external fun nativeEmbedderDestroy(handle: Long)
}
