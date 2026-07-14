#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

// Concurrency stress test for HNSWIndex.
//
// The former implementation mutated shared visited-tracking state under a
// shared (reader) lock, so concurrent knnSearch calls raced and could
// return wrong/short result lists. The visited-list pool gives each search
// its own buffer; this test hammers the index from many reader threads
// (plus a writer) and checks results stay sane. (MinGW lacks TSan, so this
// is a stress test with assertions rather than a sanitizer run.)

#include "hnsw_index.hpp"

#include <thread>
#include <atomic>
#include <random>
#include <vector>

using namespace edgevdb;

namespace {

std::vector<float> randomUnitVector(std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(EMBEDDING_DIM);
    float norm = 0.0f;
    for (auto& x : v) { x = dist(rng); norm += x * x; }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
}

} // namespace

TEST_CASE("Concurrent readers with one writer") {
    HNSWIndex index(16, 100, 32);
    std::mt19937 rng(7);

    const int INITIAL = 2000;
    std::vector<std::vector<float>> vectors;
    vectors.reserve(INITIAL);
    for (int i = 0; i < INITIAL; i++) {
        vectors.push_back(randomUnitVector(rng));
        index.insert(static_cast<uint64_t>(i + 1), vectors.back().data());
    }

    constexpr int READERS = 8;
    constexpr int QUERIES_PER_READER = 200;
    std::atomic<int> failures{0};
    std::atomic<int> completed{0};

    auto reader = [&](int seed) {
        std::mt19937 local_rng(seed);
        std::uniform_int_distribution<int> pick(0, INITIAL - 1);
        for (int q = 0; q < QUERIES_PER_READER; q++) {
            int idx = pick(local_rng);
            auto results = index.knnSearch(vectors[idx].data(), 10);
            // Sanity: non-empty, valid ids, distances sorted ascending.
            if (results.empty()) { failures++; continue; }
            bool sorted = true;
            for (size_t i = 1; i < results.size(); i++) {
                if (results[i].second < results[i - 1].second - 1e-5f) sorted = false;
            }
            if (!sorted) failures++;
            // Self-query must find itself (it was never removed).
            if (results[0].first != static_cast<uint64_t>(idx + 1) &&
                results[0].second > 0.05f) {
                failures++;
            }
            completed++;
        }
    };

    auto writer = [&]() {
        std::mt19937 local_rng(999);
        for (int i = 0; i < 200; i++) {
            auto v = randomUnitVector(local_rng);
            index.insert(static_cast<uint64_t>(INITIAL + i + 1), v.data());
        }
    };

    std::vector<std::thread> threads;
    threads.emplace_back(writer);
    for (int r = 0; r < READERS; r++) {
        threads.emplace_back(reader, 1000 + r);
    }
    for (auto& t : threads) t.join();

    CHECK(failures.load() == 0);
    CHECK(completed.load() == READERS * QUERIES_PER_READER);
    CHECK(index.size() == INITIAL + 200);
}

TEST_CASE("Compaction removes tombstones and preserves live search") {
    HNSWIndex index(16, 100, 32);
    std::mt19937 rng(11);

    const int N = 600;
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < N; i++) {
        vectors.push_back(randomUnitVector(rng));
        index.insert(static_cast<uint64_t>(i + 1), vectors.back().data());
    }

    // Delete >30% (above the compaction floor of 128) to trigger rebuild.
    for (int i = 0; i < 250; i++) {
        CHECK(index.remove(static_cast<uint64_t>(i + 1)));
    }
    CHECK(index.size() == N - 250);

    // Every live vector must still be findable as its own nearest neighbor.
    int found = 0;
    for (int i = 250; i < N; i++) {
        auto results = index.knnSearch(vectors[i].data(), 1);
        if (!results.empty() && results[0].first == static_cast<uint64_t>(i + 1)) found++;
    }
    CHECK(found >= (N - 250) * 95 / 100); // ≥95% self-recall after compaction

    // Deleted ids must never be returned.
    auto results = index.knnSearch(vectors[0].data(), 20);
    for (const auto& [id, dist] : results) {
        CHECK(id > 250);
    }
}
