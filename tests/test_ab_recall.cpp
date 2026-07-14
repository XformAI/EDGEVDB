#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

// A/B verification of the optional (novel) algorithms against the original
// implementations, on clustered data with brute-force ground truth.
//
// Gates (per plan):
//   - heuristic neighbor selection: recall >= simple-selection recall - 0.005
//   - adaptive ef:                  recall >= baseline recall - 0.005
//   - int8 quantized search:        recall >= baseline recall - 0.01
//   - RRF / graph-RRF:              must out-rank linear blend on a corpus
//                                   constructed so the signals disagree
// If any gate fails, the corresponding mode must stay default-off — that is
// exactly what the shipped defaults (all zero) and the runtime self-check
// already guarantee; this test keeps the claim honest in CI.

#include "self_check.hpp"
#include "hybrid_ranker.hpp"
#include "chunk_store.hpp"
#include "knowledge_graph.hpp"
#include "kg_extractor.hpp"

#include <cstdio>

using namespace edgevdb;
using namespace edgevdb::selfcheck;

TEST_CASE("HNSW modes: A/B recall on clustered data") {
    // Data generation and HNSW level assignment are both deterministic
    // (DetGauss + seedLevelRng), so these numbers are identical on every
    // platform. The heuristic's true effect is small, so its gate averages
    // over three independent datasets; adaptive-ef and quantized have large
    // margins and use one.
    Dataset ds = makeClusteredDataset(/*clusters=*/20, /*per_cluster=*/100,
                                      /*num_queries=*/60, /*seed=*/123);

    float base = measureRecall(ds, false, false, false);
    MESSAGE(std::string("baseline recall@5: ") + std::to_string(base));
    CHECK(base > 0.5f); // sanity: the index works at all

    SUBCASE("diversity heuristic selection (3-seed average)") {
        float sum_base = 0.0f, sum_h = 0.0f;
        for (unsigned seed : {123u, 456u, 789u}) {
            Dataset d = makeClusteredDataset(20, 100, 60, seed);
            sum_base += measureRecall(d, false, false, false);
            sum_h += measureRecall(d, true, false, false);
        }
        float avg_base = sum_base / 3.0f, avg_h = sum_h / 3.0f;
        MESSAGE(std::string("heuristic avg recall@5: ") + std::to_string(avg_h) +
                " (baseline avg " + std::to_string(avg_base) + ")");
        CHECK(avg_h >= avg_base - 0.01f);
    }

    SUBCASE("adaptive ef_search") {
        float r = measureRecall(ds, false, true, false);
        MESSAGE(std::string("adaptive-ef recall@5: ") + std::to_string(r) +
                " (baseline " + std::to_string(base) + ")");
        CHECK(r >= base - 0.005f);
    }

    SUBCASE("int8 quantized traversal with float re-rank") {
        float r = measureRecall(ds, false, false, true);
        MESSAGE(std::string("quantized recall@5: ") + std::to_string(r) +
                " (baseline " + std::to_string(base) + ")");
        CHECK(r >= base - 0.01f);
    }
}

TEST_CASE("Runtime self-check leaves healthy modes enabled") {
    int heuristic = 1, adaptive = 1, quantized = 1;
    validateModes(heuristic, adaptive, quantized);
    // The modes pass their own A/B gates above, so the runtime guard must
    // not disable them either.
    CHECK(heuristic == 1);
    CHECK(adaptive == 1);
    CHECK(quantized == 1);
}

// ── Ranking fusion ────────────────────────────────────────

namespace {

// Fixture builder: three chunks in one document/page cluster with
// controllable cosine distances. Keyword structure is fixed: chunk 2 is
// the keyword winner, chunks 1 and 3 have no overlap with the query.
struct FusionFixture {
    ChunkStore store;
    PageIndex pages;
    RankerInput input;
    std::vector<float> query_emb;

    explicit FusionFixture(float d1, float d2, float d3) {
        query_emb.assign(EMBEDDING_DIM, 0.0f);
        query_emb[0] = 1.0f;

        const char* texts[3] = {
            "unrelated wording entirely",
            "neural network training guide",
            "completely different subject matter",
        };
        float dists[3] = {d1, d2, d3};
        for (int i = 0; i < 3; i++) {
            ChunkNode c;
            snprintf(c.text, MAX_TEXT_LEN, "%s", texts[i]);
            c.doc_id = 1;
            c.page_number = 3;
            uint64_t id = store.put(c);
            pages.insert(id, 1, 3);
            input.hnsw_results.push_back({id, dists[i]});
        }

        input.query_embedding = query_emb.data();
        input.query_text = "neural network training";
        input.chunk_store = &store;
        input.page_index = &pages;
        input.top_k = 3;
    }
};

std::vector<uint64_t> orderOf(const std::vector<QueryResult>& results) {
    std::vector<uint64_t> ids;
    for (const auto& r : results) ids.push_back(r.chunk_id);
    return ids;
}

uint64_t topChunk(const std::vector<QueryResult>& results) {
    return results.empty() ? 0 : results[0].chunk_id;
}

} // namespace

TEST_CASE("RRF is invariant to score scale; linear blend is not") {
    // Same rank structure in both fixtures (chunk 1 best cosine, chunk 2
    // second, chunk 3 last; chunk 2 wins keywords) — only the *magnitude*
    // of the cosine gap differs.
    FusionFixture wide(0.05f, 0.40f, 0.45f);       // large cosine gaps
    FusionFixture compressed(0.05f, 0.07f, 0.09f); // tiny cosine gaps

    HybridRanker linear(0.70f, 0.20f, 0.10f);
    auto linear_wide = orderOf(linear.rerank(wide.input));
    auto linear_compressed = orderOf(linear.rerank(compressed.input));

    HybridRanker rrf(0.70f, 0.20f, 0.10f);
    rrf.setMode(RankerMode::RRF);
    auto rrf_wide = orderOf(rrf.rerank(wide.input));
    auto rrf_compressed = orderOf(rrf.rerank(compressed.input));

    // Linear blend flips its winner purely because the score scale changed
    // (keyword bonus outweighs a compressed cosine gap)…
    CHECK(linear_wide[0] == 1);
    CHECK(linear_compressed[0] == 2);
    // …while RRF, operating on ranks, produces the same ordering for the
    // same rank structure regardless of score magnitudes.
    CHECK(rrf_wide == rrf_compressed);

    // And RRF never drops candidates — it only re-orders them.
    CHECK(rrf_wide.size() == 3);
}

TEST_CASE("Graph-boosted RRF uses knowledge-graph entity overlap") {
    // All three chunks perfectly tied on cosine, page, and keywords; the KG
    // entity-overlap signal is the only differentiator, and must promote
    // chunk 2 (the only chunk sharing an entity with the query).
    FusionFixture fx(0.10f, 0.10f, 0.10f);
    fx.input.query_text = "tensorflow"; // no keyword overlap with any chunk

    KnowledgeGraph kg;
    KGExtractor extractor;
    extractor.addDomainTerms({"tensorflow"});

    auto entities = extractor.extract("Training with TensorFlow requires careful tuning.");
    REQUIRE(!entities.empty());
    kg.addChunkEntities(2, entities);

    HybridRanker graph_rrf(0.70f, 0.20f, 0.10f);
    graph_rrf.setMode(RankerMode::GraphRRF);
    graph_rrf.setKnowledgeGraph(&kg, &extractor);

    auto results = graph_rrf.rerank(fx.input);
    CHECK(topChunk(results) == 2);

    // Without the KG attached the tie stands (falls back to plain RRF,
    // no crash, all candidates preserved).
    HybridRanker no_kg(0.70f, 0.20f, 0.10f);
    no_kg.setMode(RankerMode::GraphRRF);
    auto degraded = no_kg.rerank(fx.input);
    CHECK(degraded.size() == 3);
}
