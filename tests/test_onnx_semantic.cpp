#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

// Semantic verification of real ONNX inference.
//
// These tests only run when the MiniLM model, the vocab, and an ONNX
// Runtime shared library are all available (set EDGEVDB_ORT_LIBRARY or put
// onnxruntime on the default library search path, and place model.onnx /
// vocab.txt under models/). Otherwise they pass trivially with a message.
//
// The semantic assertions are chosen so the deterministic hash fallback
// CANNOT pass them by construction: hash embeddings have no notion of
// synonymy, so a paraphrase ("kitten"/"cat") shares almost no tokens with
// the anchor while the distractor is engineered to share several.

#include "embedder.hpp"
#include "edgevdb/vectordb.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace edgevdb;

namespace {

std::string repoPath(const std::string& rel) {
    // Tests run from build/desktop-*/tests; walk up to the repo root.
    for (const char* prefix : {"", "../", "../../", "../../../"}) {
        std::string p = std::string(prefix) + rel;
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return rel;
}

float cosine(const float* a, const float* b) {
    float dot = 0;
    for (size_t i = 0; i < EMBEDDING_DIM; i++) dot += a[i] * b[i];
    return dot; // embeddings are L2-normalised
}

} // namespace

TEST_CASE("ONNX embedder: real semantic inference") {
    std::string model = repoPath("models/model.onnx");
    std::string vocab = repoPath("models/vocab.txt");

    OnnxEmbedder embedder;
    REQUIRE(embedder.initialize(model, vocab));

    if (!embedder.isSemantic()) {
        MESSAGE("ONNX Runtime or model unavailable - semantic checks skipped "
                "(hash fallback active). Set EDGEVDB_ORT_LIBRARY and provide "
                "models/model.onnx to run them.");
        return;
    }

    float anchor[EMBEDDING_DIM], paraphrase[EMBEDDING_DIM], distractor[EMBEDDING_DIM];

    SUBCASE("paraphrase beats lexical-overlap distractor") {
        // Paraphrase shares essentially no tokens with the anchor; the
        // distractor shares several ("the", "sat", "on", "a"). A hash
        // embedder would rank the distractor closer; a semantic model
        // must rank the paraphrase closer.
        REQUIRE(embedder.embed("The cat sat on the mat.", anchor));
        REQUIRE(embedder.embed("A kitten was resting upon a rug.", paraphrase));
        REQUIRE(embedder.embed("The committee sat on a proposal for months.", distractor));

        float sim_para = cosine(anchor, paraphrase);
        float sim_dist = cosine(anchor, distractor);
        MESSAGE(std::string("sim(paraphrase)=") + std::to_string(sim_para) +
                "  sim(distractor)=" + std::to_string(sim_dist));
        CHECK(sim_para > sim_dist);
        CHECK(sim_para > 0.4f); // paraphrases should be clearly related
    }

    SUBCASE("topical clustering") {
        REQUIRE(embedder.embed("How do I train a neural network?", anchor));
        REQUIRE(embedder.embed("What is the best way to fit a deep learning model?", paraphrase));
        REQUIRE(embedder.embed("What is the best recipe for chocolate cake?", distractor));

        CHECK(cosine(anchor, paraphrase) > cosine(anchor, distractor));
    }

    SUBCASE("full RAG pipeline through the public C API") {
        // End-to-end: open DB → insert_text (auto-embed) over a small
        // multi-topic corpus → query_text → the semantically right chunk
        // must rank first and the assembled context must contain it.
        EvdbEmbedder* e = evdb_embedder_create(model.c_str(), vocab.c_str(), 2);
        REQUIRE(e != nullptr);
        REQUIRE(evdb_embedder_is_semantic(e) == 1);

        EvdbConfig config;
        evdb_default_config(&config);
        config.storage_dir = "rag_e2e_tmp";
        EvdbHandle* db = evdb_open(&config);
        REQUIRE(db != nullptr);

        const char* corpus[] = {
            "Gradient descent updates model weights by following the negative gradient of the loss.",
            "The French Revolution began in 1789 and transformed European politics.",
            "Photosynthesis converts sunlight, water and carbon dioxide into glucose in plant cells.",
            "A balanced diet for marathon runners emphasises complex carbohydrates and hydration.",
            "Quantum entanglement links particle states across arbitrary distances.",
        };
        for (uint32_t i = 0; i < 5; i++) {
            uint64_t cid = 0;
            REQUIRE(evdb_insert_text(db, e, corpus[i], i + 1, 0, &cid) == EVDB_OK);
        }

        // Query phrased with minimal lexical overlap with the target chunk.
        EvdbQueryHandle* q = evdb_query_text(db, e, "how do plants make food from light", 3, 0);
        REQUIRE(q != nullptr);
        REQUIRE(evdb_result_count(q) >= 1);
        std::string top = evdb_result_text(q, 0);
        CHECK(top.find("Photosynthesis") != std::string::npos);

        std::string context = evdb_result_context_string(q);
        CHECK(context.find("Photosynthesis") != std::string::npos);
        evdb_query_free(q);

        // Second query for a different topic must retrieve differently.
        q = evdb_query_text(db, e, "optimising a neural model during training", 3, 0);
        REQUIRE(q != nullptr);
        top = evdb_result_text(q, 0);
        CHECK(top.find("Gradient descent") != std::string::npos);
        evdb_query_free(q);

        evdb_close(db);
        evdb_embedder_destroy(e);
        for (const char* f : {"rag_e2e_tmp/chunks.bin", "rag_e2e_tmp/hnsw.bin",
                              "rag_e2e_tmp/page.bin", "rag_e2e_tmp/kg.bin",
                              "rag_e2e_tmp/objects.bin", "rag_e2e_tmp/relations.bin"}) {
            std::remove(f);
        }
    }

    SUBCASE("deterministic and normalised") {
        float a[EMBEDDING_DIM], b[EMBEDDING_DIM];
        REQUIRE(embedder.embed("determinism check", a));
        REQUIRE(embedder.embed("determinism check", b));
        float norm = 0, diff = 0;
        for (size_t i = 0; i < EMBEDDING_DIM; i++) {
            norm += a[i] * a[i];
            diff += (a[i] - b[i]) * (a[i] - b[i]);
        }
        CHECK(std::abs(norm - 1.0f) < 1e-3f);
        CHECK(diff < 1e-8f);
    }
}
