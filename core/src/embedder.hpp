#pragma once

#include "tokenizer.hpp"
#include "schema.hpp"
#include "log.hpp"

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cmath>

// Forward declare ORT types to avoid header dependency in this public-ish
// header; embedder.cpp includes the vendored onnxruntime_c_api.h.
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtApi;

namespace edgevdb {

inline void l2Normalize(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (size_t i = 0; i < dim; i++) vec[i] /= norm;
    }
}

// Which implementation actually produces embeddings.
// HashFallback is deterministic and useful for tests/prototyping, but it is
// NOT semantic — do not use it for production similarity search.
enum class EmbedderBackend {
    HashFallback = 0,
    Onnx = 1,
};

// Text embedder. At initialize() it attempts to load ONNX Runtime
// dynamically (no link-time dependency): the shared library named by the
// EDGEVDB_ORT_LIBRARY environment variable, or the platform-default name
// (onnxruntime.dll / libonnxruntime.so / libonnxruntime.dylib). When the
// runtime and the model load successfully, embed() runs real transformer
// inference with attention-masked mean pooling and L2 normalisation.
// Otherwise it falls back to the deterministic hash embedder and warns.
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
    EmbedderBackend backend() const { return backend_; }
    // True only when a real semantic model (ONNX) is doing the embedding.
    bool isSemantic() const { return backend_ == EmbedderBackend::Onnx; }
    void shutdown();

private:
    OrtEnv* env_;
    OrtSession* session_;
    OrtSessionOptions* session_options_;
    OrtMemoryInfo* memory_info_;
    const OrtApi* ort_ = nullptr;
    void* ort_lib_ = nullptr;
    std::vector<std::string> input_names_;
    std::string output_name_;
    std::unique_ptr<WordPieceTokenizer> tokenizer_;
    bool initialized_;
    EmbedderBackend backend_ = EmbedderBackend::HashFallback;

    bool tryInitOnnx(const std::string& model_path, int num_threads);
    bool embedOnnx(const std::string& text, float* output);
    void releaseOnnx();

    // Deterministic fallback when ONNX Runtime is not available.
    void generateFallbackEmbedding(const std::string& text, float* output);
};

} // namespace edgevdb
