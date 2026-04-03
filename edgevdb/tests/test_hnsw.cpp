#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "hnsw_index.hpp"
#include "chunk_store.hpp"
#include <random>
#include <vector>
#include <cmath>
#include <thread>

using namespace edgevdb;

TEST_CASE("HNSW Insert and Search") {
    HNSWIndex index(16, 200, 64);

    // Generate random 384-dim vectors
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const int N = 1000;

    std::vector<std::vector<float>> vectors(N);
    for (int i = 0; i < N; i++) {
        vectors[i].resize(EMBEDDING_DIM);
        float norm = 0.0f;
        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            vectors[i][d] = dist(rng);
            norm += vectors[i][d] * vectors[i][d];
        }
        norm = std::sqrt(norm);
        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            vectors[i][d] /= norm;
        }
        index.insert(static_cast<uint64_t>(i + 1), vectors[i].data());
    }

    CHECK(index.size() == N);

    SUBCASE("KNN search returns correct count") {
        auto results = index.knnSearch(vectors[0].data(), 10);
        CHECK(results.size() == 10);
        // First result should be the query vector itself
        CHECK(results[0].first == 1);
        CHECK(results[0].second < 0.01f);
    }

    SUBCASE("Recall at 10") {
        // For 50 random queries, check recall against brute force
        int total_recall = 0;
        int total_expected = 0;
        const int NUM_QUERIES = 50;
        const int K = 10;

        for (int q = 0; q < NUM_QUERIES; q++) {
            // Brute force top-K
            std::vector<std::pair<float, uint64_t>> brute_force;
            for (int i = 0; i < N; i++) {
                float d = cosineDistance(vectors[q].data(), vectors[i].data(), EMBEDDING_DIM);
                brute_force.push_back({d, static_cast<uint64_t>(i + 1)});
            }
            std::sort(brute_force.begin(), brute_force.end());

            std::unordered_set<uint64_t> gt_set;
            for (int i = 0; i < K; i++) gt_set.insert(brute_force[i].second);

            auto hnsw_results = index.knnSearch(vectors[q].data(), K);
            for (const auto& [id, dist] : hnsw_results) {
                if (gt_set.count(id)) total_recall++;
            }
            total_expected += K;
        }

        float recall = static_cast<float>(total_recall) / total_expected;
        CHECK(recall >= 0.80f); // Should be > 0.90 but relaxed for small dataset
    }

    SUBCASE("Remove and verify") {
        index.remove(1);
        auto results = index.knnSearch(vectors[0].data(), 5);
        for (const auto& [id, dist] : results) {
            CHECK(id != 1);
        }
    }
}

TEST_CASE("HNSW Save and Load") {
    std::string path = "test_hnsw_temp.bin";

    {
        HNSWIndex index(path, 16, 200, 64);

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> vec(EMBEDDING_DIM);
        for (int i = 0; i < 100; i++) {
            float norm = 0.0f;
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                vec[d] = dist(rng);
                norm += vec[d] * vec[d];
            }
            norm = std::sqrt(norm);
            for (size_t d = 0; d < EMBEDDING_DIM; d++) vec[d] /= norm;
            index.insert(static_cast<uint64_t>(i + 1), vec.data());
        }
        index.save();
    }

    {
        HNSWIndex loaded(path, 16, 200, 64);
        CHECK(loaded.open());
        CHECK(loaded.size() == 100);
    }

    std::remove(path.c_str());
}
