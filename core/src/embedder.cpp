#include "embedder.hpp"
#include <cstring>
#include <numeric>
#include <functional>

namespace edgevdb {

OnnxEmbedder::OnnxEmbedder()
    : env_(nullptr), session_(nullptr), session_options_(nullptr),
      memory_info_(nullptr), initialized_(false) {}

OnnxEmbedder::~OnnxEmbedder() {
    shutdown();
}

bool OnnxEmbedder::initialize(const std::string& model_path, const std::string& vocab_path,
                               int num_threads) {
    (void)num_threads;

    // Load tokenizer
    tokenizer_ = std::make_unique<WordPieceTokenizer>();
    if (!tokenizer_->loadVocab(vocab_path)) {
        EVDB_LOG_ERROR("Embedder: Failed to load vocab from %s", vocab_path.c_str());
        return false;
    }

    // NOTE: Full ONNX Runtime integration requires linking against the ORT shared library.
    // This implementation provides a deterministic fallback embedding when ORT is not available.
    // When ORT is available, replace generateFallbackEmbedding with actual ORT inference.

    EVDB_LOG_INFO("Embedder: Initialized with vocab from %s", vocab_path.c_str());
    EVDB_LOG_INFO("Embedder: Model path: %s (ORT linking required for real inference)", model_path.c_str());
    initialized_ = true;
    return true;
}

void OnnxEmbedder::generateFallbackEmbedding(const std::string& text, float* output) {
    // Deterministic hash-based embedding for testing without ONNX Runtime
    // This produces consistent embeddings where similar texts get similar vectors
    std::memset(output, 0, EMBEDDING_DIM * sizeof(float));

    if (text.empty()) {
        // Zero vector for empty text, then normalize
        output[0] = 1.0f;
        return;
    }

    // Use token IDs to generate a deterministic embedding
    if (tokenizer_ && tokenizer_->isLoaded()) {
        auto encoded = tokenizer_->encode(text, 128);
        // Simple bag-of-words style embedding using token IDs
        for (int i = 0; i < encoded.actual_length; i++) {
            int64_t tid = encoded.input_ids[i];
            if (tid == WordPieceTokenizer::PAD_ID) continue;
            // Hash token ID to dimensions
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                // Deterministic pseudo-random contribution
                uint64_t h = static_cast<uint64_t>(tid) * 2654435761ULL + d * 40503ULL;
                h = (h ^ (h >> 16)) * 0x45d9f3b;
                h = (h ^ (h >> 16));
                float val = static_cast<float>(static_cast<int32_t>(h & 0xFFFF) - 32768) / 32768.0f;
                output[d] += val / static_cast<float>(encoded.actual_length);
            }
        }
    } else {
        // Simple character-based hash
        for (size_t i = 0; i < text.size(); i++) {
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                uint64_t h = static_cast<uint64_t>(text[i]) * 2654435761ULL + d * 40503ULL + i * 1000003ULL;
                h = (h ^ (h >> 16)) * 0x45d9f3b;
                float val = static_cast<float>(static_cast<int32_t>(h & 0xFFFF) - 32768) / 32768.0f;
                output[d] += val / static_cast<float>(text.size());
            }
        }
    }

    // L2 normalize
    l2Normalize(output, EMBEDDING_DIM);
}

bool OnnxEmbedder::embed(const std::string& text, float* output_embedding_384) {
    if (!initialized_) {
        EVDB_LOG_ERROR("Embedder: Not initialized");
        return false;
    }

    // TODO: When ONNX Runtime is linked, use actual model inference here:
    // 1. tokenizer_->encode(text, 128) → input_ids, attention_mask, token_type_ids
    // 2. Create OrtValue inputs
    // 3. Run session
    // 4. Mean pooling over non-padding positions
    // 5. L2 normalize

    // For now, use deterministic fallback
    generateFallbackEmbedding(text, output_embedding_384);
    return true;
}

bool OnnxEmbedder::embedBatch(const std::vector<std::string>& texts,
                               std::vector<std::array<float, 384>>& out) {
    if (!initialized_) return false;

    out.resize(texts.size());
    for (size_t i = 0; i < texts.size(); i++) {
        if (!embed(texts[i], out[i].data())) return false;
    }
    return true;
}

void OnnxEmbedder::shutdown() {
    // Release ORT handles in correct order when ORT is linked
    // OrtReleaseSession, OrtReleaseSessionOptions, OrtReleaseMemoryInfo, OrtReleaseEnv
    session_ = nullptr;
    session_options_ = nullptr;
    memory_info_ = nullptr;
    env_ = nullptr;
    tokenizer_.reset();
    initialized_ = false;
}

} // namespace edgevdb
