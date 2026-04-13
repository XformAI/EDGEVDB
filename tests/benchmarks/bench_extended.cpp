/**
 * @file bench_extended.cpp
 * @brief Extended benchmarks: 15k build, recall@5, recall@10, hybrid ranker recall
 */
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>

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

static float cosine_dist(const float* a, const float* b, size_t dim) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 0) ? (1.0f - dot / denom) : 1.0f;
}

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // ========================================================
    // Part 1: Build benchmark at 50k
    // ========================================================
    std::cout << "=== Extended Build Benchmark (15k) ===" << std::endl;

    int ext_sizes[] = {5000, 15000};
    for (int N : ext_sizes) {
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

    // ========================================================
    // Part 2: Query latency at 50k
    // ========================================================
    std::cout << "\n=== Query Latency at 15k ===" << std::endl;
    {
        const int N = 15000;
        const int QUERIES = 100;

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

        std::cout << "Inserting " << N << " chunks..." << std::endl;
        for (int i = 0; i < N; i++) {
            ChunkNode chunk;
            snprintf(chunk.text, sizeof(chunk.text), "Chunk %d: sample text for benchmark.", i);
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

        double total_ms = 0, min_ms = 1e9, max_ms = 0;
        for (int q = 0; q < QUERIES; q++) {
            float qemb[384];
            float norm = 0;
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                qemb[d] = dist(rng);
                norm += qemb[d] * qemb[d];
            }
            norm = std::sqrt(norm);
            for (size_t d = 0; d < EMBEDDING_DIM; d++) qemb[d] /= norm;

            CombinedQuery cq;
            cq.text_query = "benchmark query";
            cq.top_k = 5;

            auto t1 = std::chrono::high_resolution_clock::now();
            auto result = engine.query(cq, qemb);
            auto t2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
        }

        std::cout << "Avg latency: " << std::fixed << std::setprecision(2) << (total_ms / QUERIES) << " ms" << std::endl;
        std::cout << "Min latency: " << min_ms << " ms" << std::endl;
        std::cout << "Max latency: " << max_ms << " ms" << std::endl;
        bool pass = (total_ms / QUERIES) < 100.0;
        std::cout << "Target (<100ms): " << (pass ? "PASS" : "FAIL") << std::endl;
    }

    // ========================================================
    // Part 3: Recall@5 and Recall@10 (pure HNSW cosine)
    // ========================================================
    std::cout << "\n=== Recall Benchmark (Pure HNSW Cosine) ===" << std::endl;
    {
        const int N = 10000;
        const int NUM_QUERIES = 100;

        // Build dataset
        std::vector<std::vector<float>> vectors(N);
        HNSWIndex index(16, 200, 64);

        for (int i = 0; i < N; i++) {
            vectors[i].resize(EMBEDDING_DIM);
            float norm = 0;
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                vectors[i][d] = dist(rng);
                norm += vectors[i][d] * vectors[i][d];
            }
            norm = std::sqrt(norm);
            for (size_t d = 0; d < EMBEDDING_DIM; d++) vectors[i][d] /= norm;
            index.insert(static_cast<uint64_t>(i + 1), vectors[i].data());
        }

        for (int K : {5, 10}) {
            int total_recall = 0;
            int total_expected = 0;

            for (int q = 0; q < NUM_QUERIES; q++) {
                // Brute force ground truth
                std::vector<std::pair<float, uint64_t>> brute;
                for (int i = 0; i < N; i++) {
                    float d = cosine_dist(vectors[q].data(), vectors[i].data(), EMBEDDING_DIM);
                    brute.push_back({d, static_cast<uint64_t>(i + 1)});
                }
                std::sort(brute.begin(), brute.end());

                std::unordered_set<uint64_t> gt;
                for (int i = 0; i < K; i++) gt.insert(brute[i].second);

                auto hnsw_results = index.knnSearch(vectors[q].data(), K);
                for (const auto& [id, d] : hnsw_results) {
                    if (gt.count(id)) total_recall++;
                }
                total_expected += K;
            }

            float recall = static_cast<float>(total_recall) / total_expected;
            std::cout << "Recall@" << K << " (N=" << N << ", queries=" << NUM_QUERIES << "): "
                      << std::fixed << std::setprecision(4) << recall << std::endl;
        }
    }

    // ========================================================
    // Part 4: Recall under Hybrid Ranker (alpha=0.7, beta=0.2, gamma=0.1)
    // ========================================================
    std::cout << "\n=== Recall Benchmark (Hybrid Ranker, default weights) ===" << std::endl;
    {
        const int N = 10000;
        const int NUM_QUERIES = 100;
        const int K = 5;

        ChunkStore chunk_store;
        HNSWIndex hnsw_index(16, 200, 64);
        PageIndex page_index;
        HybridRanker ranker(0.70f, 0.20f, 0.10f);

        std::vector<std::vector<float>> vectors(N);
        for (int i = 0; i < N; i++) {
            vectors[i].resize(EMBEDDING_DIM);
            float norm = 0;
            for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                vectors[i][d] = dist(rng);
                norm += vectors[i][d] * vectors[i][d];
            }
            norm = std::sqrt(norm);
            for (size_t d = 0; d < EMBEDDING_DIM; d++) vectors[i][d] /= norm;

            ChunkNode chunk;
            snprintf(chunk.text, sizeof(chunk.text), "chunk %d data", i);
            std::memcpy(chunk.embedding, vectors[i].data(), EMBEDDING_DIM * sizeof(float));
            chunk.doc_id = i / 100;
            chunk.page_number = i % 100;

            uint64_t id = chunk_store.put(chunk);
            ChunkNode stored;
            chunk_store.get(id, stored);
            hnsw_index.insert(id, stored.embedding);
            page_index.insert(id, stored.doc_id, stored.page_number);
        }

        int total_recall = 0;
        int total_expected = 0;

        for (int q = 0; q < NUM_QUERIES; q++) {
            // Brute force ground truth by cosine distance
            std::vector<std::pair<float, uint64_t>> brute;
            for (int i = 0; i < N; i++) {
                float d = cosine_dist(vectors[q].data(), vectors[i].data(), EMBEDDING_DIM);
                brute.push_back({d, static_cast<uint64_t>(i + 1)});
            }
            std::sort(brute.begin(), brute.end());
            std::unordered_set<uint64_t> gt;
            for (int i = 0; i < K; i++) gt.insert(brute[i].second);

            // HNSW + Hybrid Ranker pipeline
            auto hnsw_results = hnsw_index.knnSearch(vectors[q].data(), K * 3);
            RankerInput ri;
            ri.hnsw_results = hnsw_results;
            ri.query_text = "chunk " + std::to_string(q) + " data";
            ri.chunk_store = &chunk_store;
            ri.page_index = &page_index;
            ri.top_k = K;

            auto ranked = ranker.rerank(ri);

            for (const auto& r : ranked) {
                if (gt.count(r.chunk_id)) total_recall++;
            }
            total_expected += K;
        }

        float recall = static_cast<float>(total_recall) / total_expected;
        std::cout << "Recall@" << K << " (Hybrid, N=" << N << ", queries=" << NUM_QUERIES << "): "
                  << std::fixed << std::setprecision(4) << recall << std::endl;
        std::cout << "Note: Hybrid ranker re-ranks by 0.7*cosine + 0.2*page_proximity + 0.1*keyword" << std::endl;
        std::cout << "      Ground truth is pure cosine top-K, so hybrid recall may differ." << std::endl;
    }

    return 0;
}
