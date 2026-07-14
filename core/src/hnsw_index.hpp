#pragma once

#include "schema.hpp"
#include "chunk_store.hpp"
#include "log.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
#include <mutex>
#include <memory>
#include <string>
#include <utility>
#include <queue>
#include <random>
#include <cstdint>
#include <cmath>
#include <functional>
#include <algorithm>
#include <limits>

// SIMD includes
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define EDGEVDB_USE_NEON 1
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define EDGEVDB_USE_SSE2 1
#endif

namespace edgevdb {

// ── Distance computation ──────────────────────────────────
inline float dotProduct(const float* a, const float* b, size_t dim) {
    float result = 0.0f;

#if defined(EDGEVDB_USE_NEON)
    size_t i = 0;
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= dim; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        sum0 = vmlaq_f32(sum0, va0, vb0);
        sum1 = vmlaq_f32(sum1, va1, vb1);
    }
    sum0 = vaddq_f32(sum0, sum1);
    float32x2_t r = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
    result = vget_lane_f32(vpadd_f32(r, r), 0);
    for (; i < dim; i++) result += a[i] * b[i];
#elif defined(EDGEVDB_USE_SSE2)
    size_t i = 0;
    __m128 sum0 = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m128 va0 = _mm_loadu_ps(a + i);
        __m128 vb0 = _mm_loadu_ps(b + i);
        __m128 va1 = _mm_loadu_ps(a + i + 4);
        __m128 vb1 = _mm_loadu_ps(b + i + 4);
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(va0, vb0));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(va1, vb1));
    }
    sum0 = _mm_add_ps(sum0, sum1);
    // Horizontal sum
    __m128 shuf = _mm_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2,3,0,1));
    __m128 sums = _mm_add_ps(sum0, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    result = _mm_cvtss_f32(sums);
    for (; i < dim; i++) result += a[i] * b[i];
#else
    // Scalar fallback with 8x unrolling
    size_t i = 0;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    float s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (; i + 8 <= dim; i += 8) {
        s0 += a[i]   * b[i];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
        s4 += a[i+4] * b[i+4];
        s5 += a[i+5] * b[i+5];
        s6 += a[i+6] * b[i+6];
        s7 += a[i+7] * b[i+7];
    }
    result = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for (; i < dim; i++) result += a[i] * b[i];
#endif
    return result;
}

// Cosine distance for L2-normalized vectors: dist = 1.0 - dot(a,b)
inline float cosineDistance(const float* a, const float* b, size_t dim) {
    return 1.0f - dotProduct(a, b, dim);
}

// int8 dot product (used by the optional quantized traversal path).
inline int32_t dotProductInt8(const int8_t* a, const int8_t* b, size_t dim) {
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        s0 += static_cast<int32_t>(a[i])   * b[i];
        s1 += static_cast<int32_t>(a[i+1]) * b[i+1];
        s2 += static_cast<int32_t>(a[i+2]) * b[i+2];
        s3 += static_cast<int32_t>(a[i+3]) * b[i+3];
    }
    int32_t result = (s0 + s1) + (s2 + s3);
    for (; i < dim; i++) result += static_cast<int32_t>(a[i]) * b[i];
    return result;
}

// ── Visited-list pool ─────────────────────────────────────
// Per-query visited tracking. Each concurrent search borrows its own
// VisitedList from the pool, so shared-lock readers never write shared
// state (this replaces the former racy mutable generation counter).
struct VisitedList {
    std::vector<uint32_t> marks;
    uint32_t gen = 0;

    void prepare(size_t n) {
        if (marks.size() < n) marks.resize(n, 0);
        if (++gen == 0) { // generation wrapped: reset all marks
            std::fill(marks.begin(), marks.end(), 0);
            gen = 1;
        }
    }
    bool visited(uint32_t idx) const { return marks[idx] == gen; }
    void mark(uint32_t idx) { marks[idx] = gen; }
};

class VisitedListPool {
public:
    std::unique_ptr<VisitedList> acquire() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!free_.empty()) {
            auto vl = std::move(free_.back());
            free_.pop_back();
            return vl;
        }
        return std::make_unique<VisitedList>();
    }
    void release(std::unique_ptr<VisitedList> vl) {
        std::lock_guard<std::mutex> guard(mutex_);
        free_.push_back(std::move(vl));
    }

private:
    std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedList>> free_;
};

// RAII borrow
class VisitedListGuard {
public:
    explicit VisitedListGuard(VisitedListPool& pool)
        : pool_(pool), vl_(pool.acquire()) {}
    ~VisitedListGuard() { pool_.release(std::move(vl_)); }
    VisitedList& get() { return *vl_; }

private:
    VisitedListPool& pool_;
    std::unique_ptr<VisitedList> vl_;
};

// ── HNSW Node ─────────────────────────────────────────────
struct HNSWNode {
    uint64_t external_id;
    int level;
    std::vector<std::vector<uint32_t>> neighbors; // neighbors[layer] = internal indices
    bool deleted = false;

    HNSWNode() : external_id(0), level(0), deleted(false) {}
};

// ── HNSW Index ────────────────────────────────────────────
class HNSWIndex {
public:
    HNSWIndex();
    HNSWIndex(int M, int ef_construction, int ef_search);
    explicit HNSWIndex(const std::string& file_path,
                       int M = 16, int ef_construction = 200, int ef_search = 64);

    bool open();
    bool save();

    void insert(uint64_t external_id, const float* embedding);
    bool remove(uint64_t external_id);
    std::vector<std::pair<uint64_t, float>> knnSearch(const float* query, int k, int ef = -1) const;
    size_t size() const;
    void setEfSearch(int ef);
    // Deterministic layer assignment for reproducible builds (tests,
    // self-check benchmarks). Call before inserting.
    void seedLevelRng(unsigned seed) { rng_.seed(seed); }
    void clear();
    bool rebuildFromChunkStore(const ChunkStore& store);

    // Get embedding for an internal node
    const float* getEmbedding(uint32_t internal_idx) const;

    // ── Optional algorithm modes (defaults preserve original behavior) ──
    // Diversity-aware neighbor selection (Malkov & Yashunin, Algorithm 4).
    void setHeuristicSelection(bool enabled) { use_heuristic_ = enabled; }
    bool heuristicSelection() const { return use_heuristic_; }
    // Difficulty-adaptive ef_search (re-searches with widened beam on hard queries).
    void setAdaptiveEf(bool enabled) { adaptive_ef_ = enabled; }
    bool adaptiveEf() const { return adaptive_ef_; }
    // int8 symmetric-quantized traversal with exact float re-rank.
    void setQuantizedSearch(bool enabled);
    bool quantizedSearch() const { return quantized_; }

    // Tombstone fraction that triggers automatic graph compaction.
    static constexpr double COMPACT_THRESHOLD = 0.30;
    static constexpr size_t COMPACT_FLOOR = 128;

private:
    std::vector<HNSWNode> nodes_;
    std::vector<std::vector<float>> embeddings_; // embeddings_[internal_idx][dim]
    std::unordered_map<uint64_t, uint32_t> id_to_index_;
    uint32_t entry_point_;
    int max_layer_;
    mutable std::shared_mutex rw_mutex_;

    int M_;
    int M0_;
    int ef_construction_;
    int ef_search_;
    double mL_;
    std::string file_path_;

    std::mt19937 rng_;
    mutable VisitedListPool visited_pool_;

    size_t deleted_count_ = 0;

    bool use_heuristic_ = false;
    bool adaptive_ef_ = false;
    bool quantized_ = false;

    // Quantized shadow copies of embeddings_ (only populated when quantized_).
    std::vector<std::vector<int8_t>> codes_;
    std::vector<float> scales_;

    int randomLevel();

    using DistPair = std::pair<float, uint32_t>; // (distance, internal_idx)
    using MaxHeap = std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>>;

    // Context for one traversal: precomputed query data + distance dispatch.
    struct SearchContext {
        const float* query;
        const HNSWIndex* index;
        bool quantized;
        std::vector<int8_t> q_code;
        float q_scale;

        float distanceTo(uint32_t idx) const;
    };

    SearchContext makeContext(const float* query) const;

    MaxHeap searchLayer(const SearchContext& ctx, uint32_t entry_pt, int ef, int layer,
                        VisitedList& vl) const;

    // Original neighbor selection: closest-M truncation.
    std::vector<uint32_t> selectNeighborsSimple(const MaxHeap& candidates, int M_max) const;
    // Diversity heuristic (Algorithm 4): keeps a neighbor only if it is
    // closer to the query than to any already-selected neighbor.
    std::vector<uint32_t> selectNeighborsHeuristic(const MaxHeap& candidates, int M_max) const;
    std::vector<uint32_t> selectNeighbors(const MaxHeap& candidates, int M_max) const;

    void insertUnlocked(uint64_t external_id, const float* embedding);
    void clearUnlocked();
    void maybeCompactUnlocked();
    void quantizeVector(const float* v, std::vector<int8_t>& code, float& scale) const;
    uint32_t greedyDescend(const SearchContext& ctx, uint32_t entry, int from_layer,
                           int to_layer, VisitedList& vl) const;

    static constexpr uint32_t FILE_VERSION = 2; // v2: enforced CRC trailer
};

} // namespace edgevdb
