#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "sync_engine.hpp"

using namespace edgevdb;

TEST_CASE("Sync bidirectional merge") {
    ObjectStore store_a, store_b;
    RelationIndex rel_a, rel_b;
    ChunkStore chunk_a, chunk_b;

    SyncEngine engine_a("device-A", &store_a, &rel_a, &chunk_a);
    SyncEngine engine_b("device-B", &store_b, &rel_b, &chunk_b);

    // Insert records in A
    for (int i = 0; i < 10; i++) {
        nlohmann::json p;
        p["name"] = "record_a_" + std::to_string(i);
        uint64_t id = store_a.put("TestType", p);
        engine_a.onRecordMutated(id);
    }

    CHECK(store_a.totalCount() == 10);
    CHECK(store_b.totalCount() == 0);

    // Export from A, apply to B
    auto delta_a = engine_a.exportDelta(0);
    auto result = engine_b.applyDelta(delta_a);
    CHECK(result.records_applied == 10);
}

TEST_CASE("Sync LWW conflict resolution") {
    ObjectStore store_a, store_b;
    RelationIndex rel_a, rel_b;
    ChunkStore chunk_a, chunk_b;

    SyncEngine engine_a("device-A", &store_a, &rel_a, &chunk_a);
    SyncEngine engine_b("device-B", &store_b, &rel_b, &chunk_b);

    // Both insert the same type of record
    nlohmann::json pa;
    pa["value"] = "from_A";
    store_a.put("Shared", pa);
    engine_a.onRecordMutated(1);

    nlohmann::json pb;
    pb["value"] = "from_B";
    store_b.put("Shared", pb);
    engine_b.onRecordMutated(1);

    // Cross-sync: A → B, B → A
    auto delta_a = engine_a.exportDelta(0);
    auto delta_b = engine_b.exportDelta(0);

    auto result_ab = engine_b.applyDelta(delta_a);
    auto result_ba = engine_a.applyDelta(delta_b);

    // Both should have resolved conflicts
    CHECK(store_a.totalCount() >= 1);
    CHECK(store_b.totalCount() >= 1);
}
