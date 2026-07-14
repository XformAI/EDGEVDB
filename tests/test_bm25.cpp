#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

// BM25 lexical index + retrieval-mode tests.
//
// Verifies: BM25 ranking properties (rare-term idf boost, tf saturation,
// length normalisation), the three retrieval modes through the public C API
// (hybrid / vector-only / bm25-only), that hybrid retrieval finds exact-term
// matches vector search misses, and index scalability (50k chunks).

#include "lexical_index.hpp"
#include "edgevdb/vectordb.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

using namespace edgevdb;

TEST_CASE("BM25 ranking properties") {
    LexicalIndex idx;
    idx.addChunk(1, "the quick brown fox jumps over the lazy dog");
    idx.addChunk(2, "a fast auburn fox leaps across a sleepy hound");
    idx.addChunk(3, "stock market prices rallied after the announcement");
    idx.addChunk(4, "fox fox fox fox fox fox fox fox fox fox");
    idx.addChunk(5, "the dog slept");

    SUBCASE("term match ranks above no match") {
        auto r = idx.search("brown fox", 5);
        REQUIRE(!r.empty());
        CHECK(r[0].first == 1); // matches both terms
        for (const auto& [id, score] : r) CHECK(id != 3); // no term overlap
    }

    SUBCASE("tf saturation: keyword stuffing does not dominate") {
        auto r = idx.search("quick brown fox", 5);
        REQUIRE(r.size() >= 2);
        // Chunk 1 matches 3 distinct terms; chunk 4 repeats one term 10x.
        // BM25's tf saturation must rank multi-term coverage first.
        CHECK(r[0].first == 1);
    }

    SUBCASE("rare terms outweigh common ones (idf)") {
        idx.addChunk(6, "the the the the announcement");
        auto r = idx.search("announcement", 3);
        REQUIRE(!r.empty());
        // Shorter doc with the rare term ranks by length normalisation;
        // both docs containing it must precede any non-matching doc.
        CHECK((r[0].first == 3 || r[0].first == 6));
    }

    SUBCASE("remove works") {
        idx.removeChunk(1);
        auto r = idx.search("brown", 5);
        for (const auto& [id, s] : r) CHECK(id != 1);
    }
}

TEST_CASE("Retrieval modes through the C API") {
    auto openDb = [](int mode) {
        EvdbConfig config;
        evdb_default_config(&config);
        config.storage_dir = "bm25_mode_tmp";
        config.retrieval_mode = mode;
        return evdb_open(&config);
    };

    // Corpus: chunk with a rare exact term ("zephyrium") whose embedding is
    // deliberately far from the query embedding, plus vector-close chunks.
    float far_emb[384] = {0};
    far_emb[383] = 1.0f;
    float near_emb[384] = {0};
    near_emb[0] = 1.0f;
    float query_emb[384] = {0};
    query_emb[0] = 0.9f;
    query_emb[1] = 0.436f; // ~unit norm, close to near_emb

    // ── Hybrid mode: exact-term chunk is retrievable despite far embedding ──
    {
        EvdbHandle* db = openDb(0);
        REQUIRE(db != nullptr);
        uint64_t id_far = 0, id_near = 0;
        REQUIRE(evdb_insert_chunk(db, "the zephyrium alloy datasheet", far_emb, 1, 0, &id_far) == EVDB_OK);
        REQUIRE(evdb_insert_chunk(db, "general metallurgy overview", near_emb, 1, 1, &id_near) == EVDB_OK);
        REQUIRE(evdb_insert_chunk(db, "introduction to materials science", near_emb, 1, 2, &id_near) == EVDB_OK);

        EvdbQueryHandle* q = evdb_query_vector(db, query_emb, "zephyrium alloy", 3);
        REQUIRE(q != nullptr);
        bool found_far = false;
        for (int i = 0; i < evdb_result_count(q); i++) {
            if (evdb_result_chunk_id(q, i) == id_far) found_far = true;
        }
        CHECK(found_far); // hybrid union must surface the lexical match
        evdb_query_free(q);
        evdb_close(db);
    }

    // ── Vector-only mode: same query must NOT surface the far chunk ──
    {
        std::remove("bm25_mode_tmp/chunks.bin");
        std::remove("bm25_mode_tmp/hnsw.bin");
        EvdbHandle* db = openDb(1);
        REQUIRE(db != nullptr);
        uint64_t id_far = 0, dummy = 0;
        REQUIRE(evdb_insert_chunk(db, "the zephyrium alloy datasheet", far_emb, 1, 0, &id_far) == EVDB_OK);
        REQUIRE(evdb_insert_chunk(db, "general metallurgy overview", near_emb, 1, 1, &dummy) == EVDB_OK);
        REQUIRE(evdb_insert_chunk(db, "introduction to materials science", near_emb, 1, 2, &dummy) == EVDB_OK);

        EvdbQueryHandle* q = evdb_query_vector(db, query_emb, "zephyrium alloy", 2);
        REQUIRE(q != nullptr);
        for (int i = 0; i < evdb_result_count(q); i++) {
            CHECK(evdb_result_chunk_id(q, i) != id_far);
        }
        evdb_query_free(q);
        evdb_close(db);
    }

    // ── BM25-only: no embedding needed at query time ──
    {
        std::remove("bm25_mode_tmp/chunks.bin");
        std::remove("bm25_mode_tmp/hnsw.bin");
        EvdbHandle* db = openDb(2);
        REQUIRE(db != nullptr);
        uint64_t id_far = 0, dummy = 0;
        REQUIRE(evdb_insert_chunk(db, "the zephyrium alloy datasheet", far_emb, 1, 0, &id_far) == EVDB_OK);
        REQUIRE(evdb_insert_chunk(db, "general metallurgy overview", near_emb, 1, 1, &dummy) == EVDB_OK);

        EvdbQueryHandle* q = evdb_query_lexical(db, "zephyrium datasheet", 2);
        REQUIRE(q != nullptr);
        REQUIRE(evdb_result_count(q) >= 1);
        CHECK(evdb_result_chunk_id(q, 0) == id_far);
        CHECK(std::string(evdb_result_context_string(q)).find("zephyrium") != std::string::npos);
        evdb_query_free(q);
        evdb_close(db);
    }

    // ── Lexical index survives reopen (rebuilt from chunk store) ──
    {
        EvdbHandle* db = openDb(2);
        REQUIRE(db != nullptr);
        EvdbQueryHandle* q = evdb_query_lexical(db, "zephyrium", 2);
        REQUIRE(q != nullptr);
        CHECK(evdb_result_count(q) >= 1);
        evdb_query_free(q);
        evdb_close(db);
    }

    for (const char* f : {"bm25_mode_tmp/chunks.bin", "bm25_mode_tmp/hnsw.bin",
                          "bm25_mode_tmp/page.bin", "bm25_mode_tmp/kg.bin",
                          "bm25_mode_tmp/objects.bin", "bm25_mode_tmp/relations.bin"}) {
        std::remove(f);
    }
}

TEST_CASE("Page indexing is optional") {
    float emb_a[384] = {0}; emb_a[0] = 1.0f;
    float emb_b[384] = {0}; emb_b[0] = 0.9f; emb_b[1] = 0.436f;

    // ── Disabled: DB works fully, no page.bin is created ──
    {
        EvdbConfig config;
        evdb_default_config(&config);
        config.storage_dir = "pageopt_tmp_off";
        config.enable_page_index = 0;
        EvdbHandle* db = evdb_open(&config);
        REQUIRE(db != nullptr);

        uint64_t id1 = 0, id2 = 0;
        REQUIRE(evdb_insert_chunk(db, "alpha beta gamma", emb_a, 1, 0, &id1) == EVDB_OK);
        REQUIRE(evdb_insert_chunk(db, "delta epsilon zeta", emb_b, 1, 5, &id2) == EVDB_OK);
        REQUIRE(evdb_save(db) == EVDB_OK);

        EvdbQueryHandle* q = evdb_query_vector(db, emb_a, "alpha", 2);
        REQUIRE(q != nullptr);
        REQUIRE(evdb_result_count(q) >= 1);
        CHECK(evdb_result_chunk_id(q, 0) == id1);
        // Removal path must not crash without a page index either.
        evdb_query_free(q);
        REQUIRE(evdb_remove_chunk(db, id2) == EVDB_OK);
        evdb_close(db);

        std::ifstream page_file("pageopt_tmp_off/page.bin");
        CHECK_FALSE(page_file.is_open()); // never created
    }

    // ── Enabled (default): page.bin is created and persists ──
    {
        EvdbConfig config;
        evdb_default_config(&config);
        config.storage_dir = "pageopt_tmp_on";
        EvdbHandle* db = evdb_open(&config);
        REQUIRE(db != nullptr);
        uint64_t id = 0;
        REQUIRE(evdb_insert_chunk(db, "alpha beta gamma", emb_a, 1, 0, &id) == EVDB_OK);
        REQUIRE(evdb_save(db) == EVDB_OK);
        evdb_close(db);

        std::ifstream page_file("pageopt_tmp_on/page.bin");
        CHECK(page_file.is_open());
    }

    for (const char* d : {"pageopt_tmp_off", "pageopt_tmp_on"}) {
        for (const char* f : {"chunks.bin", "hnsw.bin", "page.bin", "kg.bin",
                              "objects.bin", "relations.bin"}) {
            std::remove((std::string(d) + "/" + f).c_str());
        }
    }
}

TEST_CASE("BM25 index scalability (50k chunks)") {
    LexicalIndex idx;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> word(0, 4999);

    // 50k chunks × 40 words from a 5k-term vocabulary.
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50000; i++) {
        std::string text;
        for (int w = 0; w < 40; w++) {
            text += "term" + std::to_string(word(rng)) + " ";
        }
        idx.addChunk(static_cast<uint64_t>(i + 1), text);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MESSAGE(std::string("BM25 build 50k chunks: ") + std::to_string(build_ms) + " ms");
    CHECK(idx.size() == 50000);
    CHECK(build_ms < 30000); // generous CI bound

    // 100 queries; must stay interactive.
    t0 = std::chrono::high_resolution_clock::now();
    int nonempty = 0;
    for (int q = 0; q < 100; q++) {
        std::string query = "term" + std::to_string(word(rng)) + " term" + std::to_string(word(rng));
        auto r = idx.search(query, 10);
        if (!r.empty()) nonempty++;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double q_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100.0;
    MESSAGE(std::string("BM25 avg query at 50k: ") + std::to_string(q_ms) + " ms");
    CHECK(nonempty == 100);
    CHECK(q_ms < 50.0);
}
