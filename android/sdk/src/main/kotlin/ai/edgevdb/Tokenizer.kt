package ai.edgevdb

import android.content.Context

/**
 * Minimal WordPiece tokenizer compatible with BERT-based models.
 *
 * Reads vocab.txt from Android assets and tokenises text into token IDs
 * suitable for all-MiniLM-L6-v2 (or any BERT tokenizer with the same vocab).
 */
class WordPieceTokenizer(context: Context, vocabAssetPath: String = "vocab.txt") {

    companion object {
        const val MAX_SEQ_LEN        = 512
        private const val UNK_TOKEN  = "[UNK]"
        private const val CLS_TOKEN  = "[CLS]"
        private const val SEP_TOKEN  = "[SEP]"
        private const val PAD_TOKEN  = "[PAD]"
        private const val MAX_WORD_LEN = 100
    }

    private val vocab: Map<String, Int>
    private val clsId: Int
    private val sepId: Int
    private val padId: Int
    private val unkId: Int

    init {
        val lines = context.assets.open(vocabAssetPath)
            .bufferedReader()
            .readLines()
        vocab = lines.mapIndexed { idx, token -> token.trim() to idx }.toMap()
        clsId = vocab[CLS_TOKEN] ?: error("CLS token not found in vocab")
        sepId = vocab[SEP_TOKEN] ?: error("SEP token not found in vocab")
        padId = vocab[PAD_TOKEN] ?: 0
        unkId = vocab[UNK_TOKEN] ?: 100
    }

    data class Encoding(
        val inputIds:      LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds:  LongArray
    )

    /**
     * Tokenise [text] and return a fixed-length encoding padded to [maxLen].
     */
    fun encode(text: String, maxLen: Int = MAX_SEQ_LEN): Encoding {
        val tokens = mutableListOf(CLS_TOKEN)
        tokens += wordpieceTokenise(basicTokenise(text))
        // Truncate to leave room for [SEP]
        if (tokens.size > maxLen - 1) tokens.subList(maxLen - 1, tokens.size).clear()
        tokens += SEP_TOKEN

        val ids = LongArray(maxLen) { padId.toLong() }
        val mask = LongArray(maxLen) { 0L }

        tokens.forEachIndexed { i, tok ->
            ids[i]  = (vocab[tok] ?: unkId).toLong()
            mask[i] = 1L
        }

        return Encoding(
            inputIds      = ids,
            attentionMask = mask,
            tokenTypeIds  = LongArray(maxLen) { 0L }
        )
    }

    // ── Basic tokenisation (lowercase + strip accents + split on whitespace/punct) ──

    private fun basicTokenise(text: String): List<String> {
        return text
            .lowercase()
            .replace(Regex("[^\\p{L}\\p{N}\\s]"), " $0 ")  // space around punctuation
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
    }

    // ── WordPiece sub-word tokenisation ───────────────────────────────

    private fun wordpieceTokenise(words: List<String>): List<String> {
        val result = mutableListOf<String>()
        for (word in words) {
            if (word.length > MAX_WORD_LEN) { result += UNK_TOKEN; continue }
            var start = 0
            var bad   = false
            val subTokens = mutableListOf<String>()
            while (start < word.length) {
                var end  = word.length
                var cur: String? = null
                while (start < end) {
                    val substr = (if (start == 0) "" else "##") + word.substring(start, end)
                    if (vocab.containsKey(substr)) { cur = substr; break }
                    end--
                }
                if (cur == null) { bad = true; break }
                subTokens += cur
                start = end
            }
            result += if (bad) listOf(UNK_TOKEN) else subTokens
        }
        return result
    }
}
