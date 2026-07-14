// Scaling benchmark: query latency, recall@10 vs brute force, and memory
// growth as the corpus grows. Run from a Release build:
//   ./build/desktop-release/tests/bench_scale [max_chunks]
#include "hnsw_index.hpp"
#include "chunk_store.hpp"

#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cstdio>
#include <cstdlib>

using namespace edgevdb;
using Clock = std::chrono::high_resolution_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    const int MAX_N = argc > 1 ? std::atoi(argv[1]) : 50000;
    // argv[2]: mode flags, e.g. "ha q" — h=heuristic selection, a=adaptive ef,
    // q=quantized search, or a number = ef_search override.
    std::string modes = argc > 2 ? argv[2] : "";
    int ef_search = argc > 3 ? std::atoi(argv[3]) : 64;
    const int NUM_QUERIES = 100;
    const int K = 10;

    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Clustered data (realistic for sentence embeddings): 100 clusters.
    const int CLUSTERS = 100;
    std::vector<std::vector<float>> centers(CLUSTERS, std::vector<float>(EMBEDDING_DIM));
    for (auto& c : centers) {
        float n = 0;
        for (auto& x : c) { x = gauss(rng); n += x * x; }
        n = std::sqrt(n);
        for (auto& x : c) x /= n;
    }
    auto makeVec = [&](int cluster) {
        std::vector<float> v(EMBEDDING_DIM);
        float n = 0;
        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            v[d] = centers[cluster][d] + 0.3f * gauss(rng);
        }
        for (auto x : v) n += x * x;
        n = std::sqrt(n);
        for (auto& x : v) x /= n;
        return v;
    };

    HNSWIndex index(16, 200, ef_search);
    if (modes.find('h') != std::string::npos) index.setHeuristicSelection(true);
    if (modes.find('a') != std::string::npos) index.setAdaptiveEf(true);
    printf("modes: heuristic=%d adaptive=%d quantized=%s ef_search=%d\n",
           index.heuristicSelection(), index.adaptiveEf(),
           modes.find('q') != std::string::npos ? "yes" : "no", ef_search);

    std::vector<std::vector<float>> all;
    all.reserve(MAX_N);

    printf("%-8s %12s %12s %10s %10s %12s\n",
           "chunks", "ins/sec", "avg_q_ms", "p95_q_ms", "recall@10", "idx_mem_MB");

    int checkpoints[] = {1000, 5000, 10000, 20000, 50000, 100000};
    int inserted = 0;
    for (int cp : checkpoints) {
        if (cp > MAX_N) break;
        int batch_start = inserted;
        auto t0 = Clock::now();
        for (; inserted < cp; inserted++) {
            all.push_back(makeVec(inserted % CLUSTERS));
            index.insert(static_cast<uint64_t>(inserted + 1), all.back().data());
        }
        double build_ms = ms(t0, Clock::now());
        double ins_per_sec = (inserted - batch_start) / (build_ms / 1000.0);
        if (modes.find('q') != std::string::npos) index.setQuantizedSearch(true);

        // Queries near cluster centers
        std::vector<double> lat;
        int hits = 0, total = 0;
        for (int q = 0; q < NUM_QUERIES; q++) {
            auto qv = makeVec(q % CLUSTERS);

            // brute force truth (on a query subsample to bound runtime)
            std::vector<std::pair<float, uint64_t>> bf;
            if (q < 20) {
                bf.reserve(inserted);
                for (int i = 0; i < inserted; i++) {
                    bf.push_back({cosineDistance(qv.data(), all[i].data(), EMBEDDING_DIM),
                                  static_cast<uint64_t>(i + 1)});
                }
                std::partial_sort(bf.begin(), bf.begin() + K, bf.end());
            }

            auto tq = Clock::now();
            auto results = index.knnSearch(qv.data(), K);
            lat.push_back(ms(tq, Clock::now()));

            if (q < 20) {
                std::unordered_set<uint64_t> truth;
                for (int i = 0; i < K; i++) truth.insert(bf[i].second);
                for (const auto& [id, dist] : results) {
                    if (truth.count(id)) hits++;
                }
                total += K;
            }
        }
        std::sort(lat.begin(), lat.end());
        double avg = 0;
        for (double v : lat) avg += v;
        avg /= lat.size();
        double p95 = lat[static_cast<size_t>(lat.size() * 0.95)];

        // Approximate index memory: embeddings + graph
        double mem_mb = inserted * (EMBEDDING_DIM * 4.0 + 16 * 4.0 * 2.2 + 64) / 1048576.0;

        printf("%-8d %12.0f %12.3f %10.3f %9.1f%% %12.1f\n",
               inserted, ins_per_sec, avg, p95, 100.0 * hits / total, mem_mb);
        fflush(stdout);
    }
    return 0;
}
