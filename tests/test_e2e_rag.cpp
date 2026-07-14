#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "edgevdb/vectordb.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir _mkdir
#endif

TEST_CASE("End-to-end C API") {
    // Create temp directory
    std::string temp_dir = "test_e2e_temp";
    #ifdef _WIN32
    _mkdir(temp_dir.c_str());
    #else
    mkdir(temp_dir.c_str(), 0755);
    #endif

    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = temp_dir.c_str();

    SUBCASE("Open and close") {
        EvdbHandle* h = evdb_open(&config);
        REQUIRE(h != nullptr);
        evdb_close(h);
    }

    SUBCASE("Insert and query chunks") {
        EvdbHandle* h = evdb_open(&config);
        REQUIRE(h != nullptr);

        // Insert chunks with pre-computed embeddings
        float emb[384];
        for (int i = 0; i < 10; i++) {
            // Generate simple embedding
            for (int d = 0; d < 384; d++) {
                emb[d] = (d == i) ? 1.0f : 0.0f;
            }
            // Normalize
            float norm = 0;
            for (int d = 0; d < 384; d++) norm += emb[d] * emb[d];
            norm = sqrtf(norm);
            if (norm > 0) for (int d = 0; d < 384; d++) emb[d] /= norm;

            uint64_t chunk_id;
            std::string text = "test chunk " + std::to_string(i);
            EvdbError err = evdb_insert_chunk(h, text.c_str(), emb, 1, i, &chunk_id);
            CHECK(err == EVDB_OK);
            CHECK(chunk_id > 0);
        }

        // Query
        for (int d = 0; d < 384; d++) emb[d] = (d == 0) ? 1.0f : 0.0f;
        float norm = 0;
        for (int d = 0; d < 384; d++) norm += emb[d] * emb[d];
        norm = sqrtf(norm);
        for (int d = 0; d < 384; d++) emb[d] /= norm;

        EvdbQueryHandle* q = evdb_query_vector(h, emb, "test chunk 0", 5);
        REQUIRE(q != nullptr);
        CHECK(evdb_result_count(q) > 0);

        const char* ctx = evdb_result_context_string(q);
        CHECK(strlen(ctx) > 0);

        evdb_query_free(q);
        evdb_close(h);
    }

    SUBCASE("Object store via C API") {
        EvdbHandle* h = evdb_open(&config);
        REQUIRE(h != nullptr);

        uint64_t obj_id;
        EvdbError err = evdb_object_put(h, "Document",
            "{\"title\":\"Test\",\"author\":\"Alice\"}", &obj_id);
        CHECK(err == EVDB_OK);
        CHECK(obj_id > 0);

        char json_buf[4096];
        err = evdb_object_get(h, obj_id, json_buf, sizeof(json_buf));
        CHECK(err == EVDB_OK);
        CHECK(strstr(json_buf, "Alice") != nullptr);

        evdb_close(h);
    }

    SUBCASE("Relations via C API") {
        EvdbHandle* h = evdb_open(&config);
        REQUIRE(h != nullptr);

        EvdbError err = evdb_relation_add(h, "authored_by", 1, 100);
        CHECK(err == EVDB_OK);

        uint64_t targets[10];
        int count = 10;
        err = evdb_relation_get_targets(h, "authored_by", 1, targets, &count);
        CHECK(err == EVDB_OK);
        CHECK(count == 1);
        CHECK(targets[0] == 100);

        evdb_close(h);
    }

    SUBCASE("Version string") {
        const char* ver = evdb_version_string();
        CHECK(strcmp(ver, "0.2.0") == 0);
    }

    // Cleanup
    #ifdef _WIN32
    // Windows: remove directory recursively
    std::string cmd = "rmdir /S /Q " + temp_dir;
    system(cmd.c_str());
    #else
    // Unix/macOS/iOS: remove directory recursively
    std::string cmd = "rm -rf " + temp_dir;
    // On iOS, system() is not available, so we skip cleanup
    #ifndef __APPLE__
    system(cmd.c_str());
    #endif
    #endif
}
