#pragma once

// Automatic fallback for optional algorithm modes.
//
// When a caller enables a novel HNSW mode (heuristic selection, adaptive
// ef, quantized search), validateModes() runs a fast micro-benchmark on
// synthetic clustered data: it measures recall@5 against brute-force ground
// truth for the baseline algorithm and for each enabled mode independently.
// Any mode that scores more than FALLBACK_TOLERANCE below baseline is
// disabled for the session (with a warning), so a regression in a new
// algorithm can never silently degrade production search quality.

#include "hnsw_index.hpp"
#include "log.hpp"

#include <random>
#include <vector>
#include <algorithm>
#include <unordered_set>

namespace edgevdb {
namespace selfcheck {

constexpr float FALLBACK_TOLERANCE = 0.01f; // allowed recall drop vs baseline

// Platform-deterministic Gaussian sampler. std::normal_distribution's
// algorithm is implementation-defined (libstdc++, libc++ and MSVC differ),
// so the same seed produced different datasets per platform and CI recall
// gates disagreed with local runs. Box-Muller over mt19937 uniforms yields
// the identical sequence everywhere.
class DetGauss {
public:
    explicit DetGauss(unsigned seed) : rng_(seed) {}
    float operator()() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        float u1, u2;
        do {
            u1 = static_cast<float>(rng_()) / static_cast<float>(std::mt19937::max());
        } while (u1 <= 1e-7f);
        u2 = static_cast<float>(rng_()) / static_cast<float>(std::mt19937::max());
        float mag = std::sqrt(-2.0f * std::log(u1));
        float ang = 6.28318530718f * u2;
        spare_ = mag * std::sin(ang);
        has_spare_ = true;
        return mag * std::cos(ang);
    }

private:
    std::mt19937 rng_;
    float spare_ = 0.0f;
    bool has_spare_ = false;
};

struct Dataset {
    std::vector<std::vector<float>> points;
    std::vector<std::vector<float>> queries;
};

inline Dataset makeClusteredDataset(int clusters = 15, int per_cluster = 20,
                                    int num_queries = 20, unsigned seed = 42) {
    Dataset ds;
    DetGauss gauss(seed);

    auto normalize = [](std::vector<float>& v) {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-8f) for (float& x : v) x /= norm;
    };

    std::vector<std::vector<float>> centers(clusters);
    for (auto& c : centers) {
        c.resize(EMBEDDING_DIM);
        for (auto& x : c) x = gauss();
        normalize(c);
    }

    for (int ci = 0; ci < clusters; ci++) {
        for (int p = 0; p < per_cluster; p++) {
            std::vector<float> v(EMBEDDING_DIM);
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                v[d] = centers[ci][d] + 0.3f * gauss();
            }
            normalize(v);
            ds.points.push_back(std::move(v));
        }
    }

    for (int q = 0; q < num_queries; q++) {
        int ci = q % clusters;
        std::vector<float> v(EMBEDDING_DIM);
        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            v[d] = centers[ci][d] + 0.3f * gauss();
        }
        normalize(v);
        ds.queries.push_back(std::move(v));
    }
    return ds;
}

inline std::vector<uint64_t> bruteForceTopK(const Dataset& ds, const std::vector<float>& query, int k) {
    std::vector<std::pair<float, uint64_t>> dists;
    dists.reserve(ds.points.size());
    for (size_t i = 0; i < ds.points.size(); i++) {
        dists.push_back({cosineDistance(query.data(), ds.points[i].data(), EMBEDDING_DIM),
                         static_cast<uint64_t>(i + 1)});
    }
    std::partial_sort(dists.begin(), dists.begin() + std::min<size_t>(k, dists.size()), dists.end());
    std::vector<uint64_t> ids;
    for (int i = 0; i < k && i < static_cast<int>(dists.size()); i++) ids.push_back(dists[i].second);
    return ids;
}

// Recall@5 of an index configuration over the dataset. The level RNG is
// seeded so the comparison between configurations is graph-for-graph
// deterministic (same layer assignments) on every platform and run.
inline float measureRecall(const Dataset& ds, bool heuristic, bool adaptive, bool quantized) {
    HNSWIndex index(16, 100, 32);
    index.seedLevelRng(0xEDB1u);
    index.setHeuristicSelection(heuristic);
    for (size_t i = 0; i < ds.points.size(); i++) {
        index.insert(static_cast<uint64_t>(i + 1), ds.points[i].data());
    }
    index.setAdaptiveEf(adaptive);
    if (quantized) index.setQuantizedSearch(true);

    constexpr int K = 5;
    int hits = 0, total = 0;
    for (const auto& q : ds.queries) {
        auto truth = bruteForceTopK(ds, q, K);
        std::unordered_set<uint64_t> truth_set(truth.begin(), truth.end());
        auto found = index.knnSearch(q.data(), K);
        for (const auto& [id, dist] : found) {
            if (truth_set.count(id)) hits++;
        }
        total += K;
    }
    return total > 0 ? static_cast<float>(hits) / static_cast<float>(total) : 0.0f;
}

// Validate the enabled modes; disable (zero) any that regress recall.
inline void validateModes(int& use_heuristic, int& adaptive_ef, int& quantized) {
    if (!use_heuristic && !adaptive_ef && !quantized) return;

    Dataset ds = makeClusteredDataset();
    float base = measureRecall(ds, false, false, false);

    auto check = [&](int& flag, bool h, bool a, bool q, const char* name) {
        if (!flag) return;
        float r = measureRecall(ds, h, a, q);
        if (r < base - FALLBACK_TOLERANCE) {
            EVDB_LOG_ERROR("SelfCheck: mode '%s' regressed recall (%.3f vs baseline %.3f) — "
                           "falling back to the original algorithm for this session",
                           name, r, base);
            flag = 0;
        } else {
            EVDB_LOG_INFO("SelfCheck: mode '%s' OK (recall %.3f vs baseline %.3f)", name, r, base);
        }
    };

    check(use_heuristic, true, false, false, "hnsw_use_heuristic_selection");
    check(adaptive_ef, false, true, false, "hnsw_adaptive_ef");
    check(quantized, false, false, true, "hnsw_quantized_search");
}

} // namespace selfcheck
} // namespace edgevdb
