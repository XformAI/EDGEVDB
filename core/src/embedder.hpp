#pragma once

#include "tokenizer.hpp"
#include "schema.hpp"
#include "log.hpp"

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cmath>

// Forward declare ORT types to avoid header dependency
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;

namespace edgevdb {

inline void l2Normalize(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (size_t i = 0; i < dim; i++) vec[i] /= norm;
    }
}

class OnnxEmbedder {
public:
    OnnxEmbedder();
    ~OnnxEmbedder();

    bool initialize(const std::string& model_path, const std::string& vocab_path,
                    int num_threads = 2);

    bool embed(const std::string& text, float* output_embedding_384);

    bool embedBatch(const std::vector<std::string>& texts,
                    std::vector<std::array<float, 384>>& out);

    bool isInitialized() const { return initialized_; }
    void shutdown();

private:
    OrtEnv* env_;
    OrtSession* session_;
    OrtSessionOptions* session_options_;
    OrtMemoryInfo* memory_info_;
    std::unique_ptr<WordPieceTokenizer> tokenizer_;
    bool initialized_;

    // Placeholder for when ONNX Runtime is not available
    void generateFallbackEmbedding(const std::string& text, float* output);
};

} // namespace edgevdb
