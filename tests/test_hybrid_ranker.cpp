#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "hybrid_ranker.hpp"
#include <cstring>

using namespace edgevdb;

TEST_CASE("Keyword overlap") {
    SUBCASE("Identical texts return 1.0") {
        float score = HybridRanker::computeKeywordOverlap("hello world", "hello world");
        CHECK(score > 0.99f);
    }

    SUBCASE("Disjoint texts return 0.0") {
        float score = HybridRanker::computeKeywordOverlap("cat dog", "airplane boat");
        CHECK(score < 0.01f);
    }

    SUBCASE("Partial overlap") {
        float score = HybridRanker::computeKeywordOverlap("machine learning model", "deep learning algorithm");
        CHECK(score > 0.0f);
        CHECK(score < 1.0f);
    }
}

TEST_CASE("Pure cosine ranking with alpha=1") {
    HybridRanker ranker(1.0f, 0.0f, 0.0f);

    ChunkStore store;
    ChunkNode c1, c2;
    std::strncpy(c1.text, "chunk one text", MAX_TEXT_LEN);
    std::strncpy(c2.text, "chunk two text", MAX_TEXT_LEN);
    c1.doc_id = 1; c1.page_number = 0;
    c2.doc_id = 1; c2.page_number = 5;

    uint64_t id1 = store.put(c1);
    uint64_t id2 = store.put(c2);

    RankerInput input;
    input.hnsw_results = {{id1, 0.1f}, {id2, 0.3f}}; // id1 closer
    input.query_text = "query";
    input.chunk_store = &store;
    input.page_index = nullptr;
    input.top_k = 2;

    auto results = ranker.rerank(input);
    CHECK(results.size() == 2);
    CHECK(results[0].chunk_id == id1); // closer cosine should rank first
    CHECK(results[0].score > results[1].score);
}
