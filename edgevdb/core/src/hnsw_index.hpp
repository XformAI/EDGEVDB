#pragma once

#include "schema.hpp"
#include "chunk_store.hpp"
#include "log.hpp"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
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
    void clear();
    bool rebuildFromChunkStore(const ChunkStore& store);

    // Get embedding for an internal node
    const float* getEmbedding(uint32_t internal_idx) const;

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

    mutable std::mt19937 rng_;
    mutable uint64_t visit_counter_;
    mutable std::vector<uint64_t> visited_gen_;

    int randomLevel();

    using DistPair = std::pair<float, uint32_t>; // (distance, internal_idx)

    // Search layer: returns candidates sorted by distance (closest first)
    std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>>
    searchLayer(const float* query, uint32_t entry_pt, int ef, int layer) const;

    // Select neighbors (simple heuristic)
    std::vector<uint32_t> selectNeighbors(
        const std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>>& candidates,
        int M_max) const;

    uint32_t computeCRC32(const uint8_t* data, size_t len) const;
};

} // namespace edgevdb
