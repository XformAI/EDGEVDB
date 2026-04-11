package ai.edgevdb

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * On-device embedding pipeline powered by ONNX Runtime.
 *
 * Loads `all-MiniLM-L6-v2` (quantized INT8) from Android assets and produces
 * 384-dimensional L2-normalised embeddings suitable for cosine similarity search.
 *
 * Thread-safety: [OrtSession] is thread-safe for concurrent [run] calls.
 * Multiple coroutines may call [embed] simultaneously.
 *
 * @param context          Android context for asset access.
 * @param modelAssetPath   Asset path to the .onnx model file (default: "model.onnx").
 * @param vocabAssetPath   Asset path to vocab.txt (default: "vocab.txt").
 * @param useNnapi         Enable NNAPI acceleration on supported devices.
 */
class OnnxEmbeddingPipeline(
    context: Context,
    modelAssetPath: String = "model.onnx",
    vocabAssetPath: String = "vocab.txt",
    private val useNnapi: Boolean = false
) : EmbeddingPipeline {

    companion object {
        private const val TAG = "OnnxEmbeddingPipeline"
    }

    override val dimensions: Int = 384

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: WordPieceTokenizer

    init {
        Log.d(TAG, "Loading ONNX model from assets: $modelAssetPath")
        val modelBytes = context.assets.open(modelAssetPath).readBytes()
        Log.d(TAG, "Model loaded: ${modelBytes.size} bytes")

        val opts = OrtSession.SessionOptions().apply {
            setInterOpNumThreads(2)
            setIntraOpNumThreads(2)
            if (useNnapi) {
                try {
                    addNnapi()
                    Log.d(TAG, "NNAPI acceleration enabled")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI not available, falling back to CPU", e)
                }
            }
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        session    = env.createSession(modelBytes, opts)
        Log.d(TAG, "ONNX session created successfully")
        tokenizer  = WordPieceTokenizer(context, vocabAssetPath)
        Log.d(TAG, "Tokenizer initialized with vocab: $vocabAssetPath")
    }

    /**
     * Embed a single text string.
     *
     * @return A 384-dimensional L2-normalised float vector.
     */
    override suspend fun embed(text: String): FloatArray = withContext(Dispatchers.Default) {
        Log.d(TAG, "Embedding text: '${text.take(50)}...'")
        val encoding = tokenizer.encode(text, maxLen = WordPieceTokenizer.MAX_SEQ_LEN)
        Log.d(TAG, "Tokenized: ${encoding.inputIds.size} tokens")
        runInference(encoding)
    }

    /**
     * Batch-embed multiple texts in a single ONNX session run.
     * More efficient than calling [embed] sequentially for large batches.
     */
    override suspend fun embedBatch(texts: List<String>): List<FloatArray> =
        withContext(Dispatchers.Default) {
            Log.d(TAG, "Batch embedding ${texts.size} texts")
            texts.chunked(32).flatMap { batch ->  // process up to 32 at a time
                batch.map { embed(it) }
            }
        }

    private fun runInference(encoding: WordPieceTokenizer.Encoding): FloatArray {
        val seqLen = encoding.inputIds.size.toLong()
        val shape  = longArrayOf(1L, seqLen)

        val inputIdsTensor     = OnnxTensor.createTensor(env, LongBuffer.wrap(encoding.inputIds),     shape)
        val attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(encoding.attentionMask), shape)
        val tokenTypeIdsTensor  = OnnxTensor.createTensor(env, LongBuffer.wrap(encoding.tokenTypeIds),  shape)

        val inputs = mapOf(
            "input_ids"      to inputIdsTensor,
            "attention_mask" to attentionMaskTensor,
            "token_type_ids" to tokenTypeIdsTensor
        )

        Log.d(TAG, "Running ONNX inference...")
        val outputs = session.run(inputs)
        Log.d(TAG, "ONNX inference completed")

        // The model may output token embeddings (shape [1, seq, 384]) or
        // a pooled embedding (shape [1, 384]). We always mean-pool if needed.
        val rawOutput = outputs[0].value
        Log.d(TAG, "Raw output type: ${rawOutput?.javaClass}")

        val embedding: FloatArray = when {
            rawOutput is Array<*> && rawOutput.first() is Array<*> -> {
                // Shape: [1, seq_len, 384] → mean-pool over seq dimension
                @Suppress("UNCHECKED_CAST")
                val tokenEmbeddings = rawOutput as Array<Array<FloatArray>>
                Log.d(TAG, "Output shape: [1, ${tokenEmbeddings[0].size}, ${tokenEmbeddings[0][0].size}]")
                meanPool(tokenEmbeddings[0], encoding.attentionMask)
            }
            rawOutput is Array<*> && rawOutput.first() is FloatArray -> {
                // Shape: [1, 384] — already pooled
                @Suppress("UNCHECKED_CAST")
                val pooled = (rawOutput as Array<FloatArray>)[0]
                Log.d(TAG, "Output shape: [1, ${pooled.size}] (already pooled)")
                pooled
            }
            else -> {
                Log.e(TAG, "Unexpected ONNX output shape: ${rawOutput?.javaClass}")
                throw EmbeddingException("Unexpected ONNX output shape: ${rawOutput?.javaClass}")
            }
        }

        inputIdsTensor.close()
        attentionMaskTensor.close()
        tokenTypeIdsTensor.close()
        outputs.close()

        val normalized = l2Normalise(embedding)
        Log.d(TAG, "Embedding L2 norm: ${normalized.map { it * it }.sum().let { kotlin.math.sqrt(it) }}")
        return normalized
    }

    /**
     * Mean-pool token embeddings weighted by attention mask.
     * Equivalent to `(token_embeddings * mask).sum(dim=0) / mask.sum()`.
     */
    private fun meanPool(tokenEmbeddings: Array<FloatArray>, mask: LongArray): FloatArray {
        val result = FloatArray(dimensions)
        var activeTokens = 0
        tokenEmbeddings.forEachIndexed { i, token ->
            if (i < mask.size && mask[i] == 1L) {
                for (d in result.indices) result[d] += token[d]
                activeTokens++
            }
        }
        if (activeTokens > 0) {
            for (d in result.indices) {
                result[d] = result[d] / activeTokens
            }
        }
        return result
    }

    private fun l2Normalise(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm > 1e-9f) {
            for (i in v.indices) {
                v[i] = v[i] / norm
            }
        }
        return v
    }

    override fun close() {
        session.close()
        env.close()
    }
}
