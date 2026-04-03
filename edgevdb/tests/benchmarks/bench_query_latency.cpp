#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>

#include "hnsw_index.hpp"
#include "chunk_store.hpp"
#include "page_index.hpp"
#include "hybrid_ranker.hpp"
#include "query_engine.hpp"
#include "kg_extractor.hpp"
#include "knowledge_graph.hpp"
#include "kg_expander.hpp"
#include "object_store.hpp"
#include "relation_index.hpp"

using namespace edgevdb;

int main() {
    std::cout << "=== EdgeVDB Query Latency Benchmark ===" << std::endl;

    const int N = 10000;   // 10k chunks
    const int QUERIES = 100;
    const int TOP_K = 5;

    // Setup components
    ChunkStore chunk_store;
    HNSWIndex hnsw_index(16, 200, 64);
    PageIndex page_index;
    HybridRanker ranker;
    KGExtractor kg_extractor;
    KnowledgeGraph kg;
    KGExpander kg_expander(&kg, &kg_extractor, &chunk_store);
    ObjectStore obj_store;
    RelationIndex rel_index;
    QueryEngine engine(&chunk_store, &hnsw_index, &page_index, &ranker,
                        &kg_extractor, &kg, &kg_expander, &obj_store, &rel_index);

    // Generate random vectors
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::cout << "Inserting " << N << " chunks..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        ChunkNode chunk;
        snprintf(chunk.text, sizeof(chunk.text), "Chunk %d: Lorem ipsum dolor sit amet.", i);
        float norm = 0.0f;
        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            chunk.embedding[d] = dist(rng);
            norm += chunk.embedding[d] * chunk.embedding[d];
        }
        norm = std::sqrt(norm);
        for (size_t d = 0; d < EMBEDDING_DIM; d++) chunk.embedding[d] /= norm;

        chunk.doc_id = i / 100;
        chunk.page_number = i % 100;

        uint64_t id = chunk_store.put(chunk);
        ChunkNode stored;
        chunk_store.get(id, stored);
        hnsw_index.insert(id, stored.embedding);
        page_index.insert(id, stored.doc_id, stored.page_number);
    }

    auto t_build = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t_build - t_start).count();
    std::cout << "Build time: " << std::fixed << std::setprecision(1) << build_ms << " ms" << std::endl;
    std::cout << "Throughput: " << std::setprecision(0) << (N / (build_ms / 1000.0)) << " chunks/sec" << std::endl;

    // Query benchmark
    std::cout << "\nRunning " << QUERIES << " queries (top_k=" << TOP_K << ")..." << std::endl;

    double total_query_ms = 0;
    double min_ms = 1e9, max_ms = 0;

    for (int q = 0; q < QUERIES; q++) {
        float query_emb[384];
        float norm = 0.0f;
        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            query_emb[d] = dist(rng);
            norm += query_emb[d] * query_emb[d];
        }
        norm = std::sqrt(norm);
        for (size_t d = 0; d < EMBEDDING_DIM; d++) query_emb[d] /= norm;

        CombinedQuery cq;
        cq.text_query = "benchmark query";
        cq.top_k = TOP_K;

        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = engine.query(cq, query_emb);
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_query_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Avg latency: " << std::fixed << std::setprecision(2) << (total_query_ms / QUERIES) << " ms" << std::endl;
    std::cout << "Min latency: " << min_ms << " ms" << std::endl;
    std::cout << "Max latency: " << max_ms << " ms" << std::endl;
    std::cout << "P50 estimate: ~" << (total_query_ms / QUERIES) << " ms" << std::endl;

    bool pass = (total_query_ms / QUERIES) < 100.0;
    std::cout << "\nTarget (<100ms): " << (pass ? "PASS ✓" : "FAIL ✗") << std::endl;

    return pass ? 0 : 1;
}
