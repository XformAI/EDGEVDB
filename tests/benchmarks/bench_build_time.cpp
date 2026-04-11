#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

#include "hnsw_index.hpp"
#include "chunk_store.hpp"

using namespace edgevdb;

int main() {
    std::cout << "=== EdgeVDB Build Time Benchmark ===" << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int sizes[] = {100, 1000, 5000, 10000};
    for (int N : sizes) {
        ChunkStore store;
        HNSWIndex index(16, 200, 64);

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            ChunkNode chunk;
            snprintf(chunk.text, sizeof(chunk.text), "Chunk %d", i);
            float norm = 0.0f;
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                chunk.embedding[d] = dist(rng);
                norm += chunk.embedding[d] * chunk.embedding[d];
            }
            norm = std::sqrt(norm);
            for (size_t d = 0; d < EMBEDDING_DIM; d++) chunk.embedding[d] /= norm;

            uint64_t id = store.put(chunk);
            ChunkNode stored;
            store.get(id, stored);
            index.insert(id, stored.embedding);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double rate = N / (ms / 1000.0);

        std::cout << "N=" << std::setw(6) << N
                  << "  Build: " << std::fixed << std::setprecision(1) << ms << " ms"
                  << "  Rate: " << std::setprecision(0) << rate << " chunks/sec"
                  << std::endl;
    }

    return 0;
}
