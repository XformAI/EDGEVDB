#include "embedder.hpp"

#include <onnxruntime/onnxruntime_c_api.h>

#include <cstring>
#include <cstdlib>
#include <numeric>
#include <functional>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace edgevdb {

namespace {

// ── Dynamic library loading (keeps ORT a runtime-optional dependency) ──

void* loadLibrary(const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* getSymbol(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void closeLibrary(void* lib) {
    if (!lib) return;
#ifdef _WIN32
    FreeLibrary(reinterpret_cast<HMODULE>(lib));
#else
    dlclose(lib);
#endif
}

void* loadOrtLibrary() {
    // Explicit override first, then platform-default names.
    const char* override_path = std::getenv("EDGEVDB_ORT_LIBRARY");
    if (override_path && override_path[0]) {
        void* lib = loadLibrary(override_path);
        if (lib) return lib;
        EVDB_LOG_ERROR("Embedder: EDGEVDB_ORT_LIBRARY set to '%s' but it failed to load",
                       override_path);
        return nullptr;
    }
#ifdef _WIN32
    const char* names[] = {"onnxruntime.dll"};
#elif defined(__APPLE__)
    const char* names[] = {"libonnxruntime.dylib", "libonnxruntime.1.dylib"};
#else
    const char* names[] = {"libonnxruntime.so", "libonnxruntime.so.1"};
#endif
    for (const char* n : names) {
        void* lib = loadLibrary(n);
        if (lib) return lib;
    }
    return nullptr;
}

#ifdef _WIN32
std::wstring toWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(len > 0 ? len - 1 : 0, L'\0');
    if (len > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], len);
    return w;
}
#endif

} // namespace

OnnxEmbedder::OnnxEmbedder()
    : env_(nullptr), session_(nullptr), session_options_(nullptr),
      memory_info_(nullptr), initialized_(false) {}

OnnxEmbedder::~OnnxEmbedder() {
    shutdown();
}

// ── ONNX Runtime initialization ───────────────────────────

bool OnnxEmbedder::tryInitOnnx(const std::string& model_path, int num_threads) {
    ort_lib_ = loadOrtLibrary();
    if (!ort_lib_) return false;

    auto get_base = reinterpret_cast<const OrtApiBase* (*)()>(getSymbol(ort_lib_, "OrtGetApiBase"));
    if (!get_base) {
        EVDB_LOG_ERROR("Embedder: loaded ORT library has no OrtGetApiBase");
        closeLibrary(ort_lib_);
        ort_lib_ = nullptr;
        return false;
    }

    const OrtApiBase* base = get_base();
    // Ask for our compiled API version first, then walk down so older
    // runtimes still work (the C API is append-only).
    for (uint32_t v = ORT_API_VERSION; v >= 17 && !ort_; v--) {
        ort_ = base->GetApi(v);
    }
    if (!ort_) {
        EVDB_LOG_ERROR("Embedder: ORT library too old (no compatible API version)");
        closeLibrary(ort_lib_);
        ort_lib_ = nullptr;
        return false;
    }

    auto fail = [this](const char* what, OrtStatus* status) {
        if (status) {
            EVDB_LOG_ERROR("Embedder: %s failed: %s", what, ort_->GetErrorMessage(status));
            ort_->ReleaseStatus(status);
        } else {
            EVDB_LOG_ERROR("Embedder: %s failed", what);
        }
        releaseOnnx();
        return false;
    };

    OrtStatus* st = ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "edgevdb", &env_);
    if (st) return fail("CreateEnv", st);

    st = ort_->CreateSessionOptions(&session_options_);
    if (st) return fail("CreateSessionOptions", st);
    st = ort_->SetIntraOpNumThreads(session_options_, num_threads > 0 ? num_threads : 2);
    if (st) return fail("SetIntraOpNumThreads", st);

#ifdef _WIN32
    std::wstring wpath = toWide(model_path);
    st = ort_->CreateSession(env_, wpath.c_str(), session_options_, &session_);
#else
    st = ort_->CreateSession(env_, model_path.c_str(), session_options_, &session_);
#endif
    if (st) return fail("CreateSession (model load)", st);

    st = ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_);
    if (st) return fail("CreateCpuMemoryInfo", st);

    // Discover the model's actual input/output names instead of assuming;
    // MiniLM exports vary in whether token_type_ids is present.
    OrtAllocator* allocator = nullptr;
    st = ort_->GetAllocatorWithDefaultOptions(&allocator);
    if (st) return fail("GetAllocatorWithDefaultOptions", st);

    size_t input_count = 0;
    st = ort_->SessionGetInputCount(session_, &input_count);
    if (st) return fail("SessionGetInputCount", st);
    if (input_count == 0 || input_count > 4) return fail("unexpected input count", nullptr);

    input_names_.clear();
    for (size_t i = 0; i < input_count; i++) {
        char* name = nullptr;
        st = ort_->SessionGetInputName(session_, i, allocator, &name);
        if (st) return fail("SessionGetInputName", st);
        input_names_.emplace_back(name);
        allocator->Free(allocator, name);
    }

    char* out_name = nullptr;
    st = ort_->SessionGetOutputName(session_, 0, allocator, &out_name);
    if (st) return fail("SessionGetOutputName", st);
    output_name_ = out_name;
    allocator->Free(allocator, out_name);

    backend_ = EmbedderBackend::Onnx;
    EVDB_LOG_INFO("Embedder: ONNX Runtime %s loaded, model '%s' (%zu inputs, output '%s')",
                  base->GetVersionString(), model_path.c_str(), input_count,
                  output_name_.c_str());
    return true;
}

void OnnxEmbedder::releaseOnnx() {
    if (ort_) {
        if (memory_info_) { ort_->ReleaseMemoryInfo(memory_info_); memory_info_ = nullptr; }
        if (session_) { ort_->ReleaseSession(session_); session_ = nullptr; }
        if (session_options_) { ort_->ReleaseSessionOptions(session_options_); session_options_ = nullptr; }
        if (env_) { ort_->ReleaseEnv(env_); env_ = nullptr; }
        ort_ = nullptr;
    }
    if (ort_lib_) {
        closeLibrary(ort_lib_);
        ort_lib_ = nullptr;
    }
    backend_ = EmbedderBackend::HashFallback;
}

bool OnnxEmbedder::initialize(const std::string& model_path, const std::string& vocab_path,
                               int num_threads) {
    // Load tokenizer
    tokenizer_ = std::make_unique<WordPieceTokenizer>();
    if (!tokenizer_->loadVocab(vocab_path)) {
        EVDB_LOG_ERROR("Embedder: Failed to load vocab from %s", vocab_path.c_str());
        return false;
    }

    // Attempt real inference; fall back to the hash embedder if ORT or the
    // model are unavailable.
    if (!tryInitOnnx(model_path, num_threads)) {
        backend_ = EmbedderBackend::HashFallback;
        // Loud, one-time honesty warning: the fallback is a deterministic
        // token-hash embedding. It is stable and fine for tests, but it has
        // no semantic meaning — similarity search over it is meaningless.
        static bool warned = false;
        if (!warned) {
            warned = true;
            EVDB_LOG_ERROR(
                "Embedder: ONNX Runtime unavailable — using the deterministic "
                "HASH-FALLBACK embedder (model '%s' is NOT loaded). This is NOT "
                "semantic inference; do not use insert_text/query_text for "
                "production search. Install ONNX Runtime (or set "
                "EDGEVDB_ORT_LIBRARY to its shared library) for real inference, "
                "or supply pre-computed embeddings via insert_chunk/query_vector.",
                model_path.c_str());
        }
    }

    EVDB_LOG_INFO("Embedder: Initialized with vocab from %s (backend=%s)",
                  vocab_path.c_str(),
                  backend_ == EmbedderBackend::Onnx ? "onnx" : "hash-fallback");
    initialized_ = true;
    return true;
}

// ── Real inference ────────────────────────────────────────

bool OnnxEmbedder::embedOnnx(const std::string& text, float* output) {
    constexpr int MAX_LEN = 128;
    auto encoded = tokenizer_->encode(text, MAX_LEN);

    const int64_t shape[2] = {1, MAX_LEN};
    const size_t bytes = MAX_LEN * sizeof(int64_t);

    auto fail = [this](const char* what, OrtStatus* status) {
        if (status) {
            EVDB_LOG_ERROR("Embedder: %s failed: %s", what, ort_->GetErrorMessage(status));
            ort_->ReleaseStatus(status);
        } else {
            EVDB_LOG_ERROR("Embedder: %s failed", what);
        }
        return false;
    };

    // Bind each model input by name.
    std::vector<OrtValue*> inputs(input_names_.size(), nullptr);
    std::vector<const char*> input_name_ptrs;
    bool ok = true;
    OrtStatus* st = nullptr;
    for (size_t i = 0; i < input_names_.size() && ok; i++) {
        input_name_ptrs.push_back(input_names_[i].c_str());
        int64_t* data = nullptr;
        if (input_names_[i] == "input_ids") data = encoded.input_ids.data();
        else if (input_names_[i] == "attention_mask") data = encoded.attention_mask.data();
        else if (input_names_[i] == "token_type_ids") data = encoded.token_type_ids.data();
        else {
            EVDB_LOG_ERROR("Embedder: unknown model input '%s'", input_names_[i].c_str());
            ok = false;
            break;
        }
        st = ort_->CreateTensorWithDataAsOrtValue(
            memory_info_, data, bytes, shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[i]);
        if (st) { ok = fail("CreateTensor", st); }
    }

    OrtValue* out_value = nullptr;
    if (ok) {
        const char* out_names[] = {output_name_.c_str()};
        st = ort_->Run(session_, nullptr,
                       input_name_ptrs.data(), inputs.data(), inputs.size(),
                       out_names, 1, &out_value);
        if (st) ok = fail("Run", st);
    }

    if (ok) {
        // Inspect output shape: [1, seq, hidden] token embeddings need
        // masked mean pooling; [1, hidden] is already pooled.
        OrtTensorTypeAndShapeInfo* info = nullptr;
        st = ort_->GetTensorTypeAndShape(out_value, &info);
        if (st) ok = fail("GetTensorTypeAndShape", st);

        size_t dim_count = 0;
        int64_t dims[4] = {0, 0, 0, 0};
        if (ok) {
            st = ort_->GetDimensionsCount(info, &dim_count);
            if (!st && dim_count <= 4) st = ort_->GetDimensions(info, dims, dim_count);
            if (st) ok = fail("GetDimensions", st);
        }
        if (info) ort_->ReleaseTensorTypeAndShapeInfo(info);

        float* out_data = nullptr;
        if (ok) {
            st = ort_->GetTensorMutableData(out_value, reinterpret_cast<void**>(&out_data));
            if (st) ok = fail("GetTensorMutableData", st);
        }

        if (ok) {
            if (dim_count == 3 && dims[2] == static_cast<int64_t>(EMBEDDING_DIM)) {
                // Masked mean pooling over token embeddings.
                const int64_t seq = dims[1];
                std::memset(output, 0, EMBEDDING_DIM * sizeof(float));
                float count = 0.0f;
                for (int64_t t = 0; t < seq && t < MAX_LEN; t++) {
                    if (encoded.attention_mask[t] == 0) continue;
                    const float* tok = out_data + t * EMBEDDING_DIM;
                    for (size_t d = 0; d < EMBEDDING_DIM; d++) output[d] += tok[d];
                    count += 1.0f;
                }
                if (count > 0.0f) {
                    for (size_t d = 0; d < EMBEDDING_DIM; d++) output[d] /= count;
                }
            } else if (dim_count == 2 && dims[1] == static_cast<int64_t>(EMBEDDING_DIM)) {
                std::memcpy(output, out_data, EMBEDDING_DIM * sizeof(float));
            } else {
                EVDB_LOG_ERROR("Embedder: unexpected output shape (%zu dims, last=%lld)",
                               dim_count, (long long)(dim_count ? dims[dim_count - 1] : 0));
                ok = false;
            }
        }
        if (ok) l2Normalize(output, EMBEDDING_DIM);
    }

    for (OrtValue* v : inputs) {
        if (v) ort_->ReleaseValue(v);
    }
    if (out_value) ort_->ReleaseValue(out_value);
    return ok;
}

// ── Fallback ──────────────────────────────────────────────

void OnnxEmbedder::generateFallbackEmbedding(const std::string& text, float* output) {
    // Deterministic hash-based embedding for testing without ONNX Runtime
    std::memset(output, 0, EMBEDDING_DIM * sizeof(float));

    if (text.empty()) {
        output[0] = 1.0f;
        return;
    }

    if (tokenizer_ && tokenizer_->isLoaded()) {
        auto encoded = tokenizer_->encode(text, 128);
        for (int i = 0; i < encoded.actual_length; i++) {
            int64_t tid = encoded.input_ids[i];
            if (tid == tokenizer_->padId()) continue;
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                uint64_t h = static_cast<uint64_t>(tid) * 2654435761ULL + d * 40503ULL;
                h = (h ^ (h >> 16)) * 0x45d9f3b;
                h = (h ^ (h >> 16));
                float val = static_cast<float>(static_cast<int32_t>(h & 0xFFFF) - 32768) / 32768.0f;
                output[d] += val / static_cast<float>(encoded.actual_length);
            }
        }
    } else {
        for (size_t i = 0; i < text.size(); i++) {
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                uint64_t h = static_cast<uint64_t>(text[i]) * 2654435761ULL + d * 40503ULL + i * 1000003ULL;
                h = (h ^ (h >> 16)) * 0x45d9f3b;
                float val = static_cast<float>(static_cast<int32_t>(h & 0xFFFF) - 32768) / 32768.0f;
                output[d] += val / static_cast<float>(text.size());
            }
        }
    }

    l2Normalize(output, EMBEDDING_DIM);
}

bool OnnxEmbedder::embed(const std::string& text, float* output_embedding_384) {
    if (!initialized_) {
        EVDB_LOG_ERROR("Embedder: Not initialized");
        return false;
    }

    if (backend_ == EmbedderBackend::Onnx) {
        if (embedOnnx(text, output_embedding_384)) return true;
        EVDB_LOG_ERROR("Embedder: ONNX inference failed for this input — using fallback");
    }

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
    releaseOnnx();
    tokenizer_.reset();
    initialized_ = false;
}

} // namespace edgevdb
