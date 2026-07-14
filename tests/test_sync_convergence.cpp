#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

// Replica convergence tests for the sync engine.
//
// The previous implementation compared wall-clock chunk timestamps against
// logical clocks and re-assigned chunk ids on apply, so replicas diverged.
// These tests verify: identical state after bidirectional exchange,
// preserved chunk ids (device-partitioned), delete propagation, and
// idempotent re-application.

#include "sync_engine.hpp"

#include <algorithm>
#include <set>
#include <cstdio>

using namespace edgevdb;

namespace {

struct Replica {
    ObjectStore objects;
    RelationIndex relations;
    ChunkStore chunks;
    SyncEngine engine;

    explicit Replica(const std::string& device_id)
        : engine(device_id, &objects, &relations, &chunks) {}

    uint64_t addChunk(const std::string& text, uint32_t doc, uint32_t page) {
        ChunkNode c;
        snprintf(c.text, MAX_TEXT_LEN, "%s", text.c_str());
        c.doc_id = doc;
        c.page_number = page;
        for (size_t d = 0; d < EMBEDDING_DIM; d++) c.embedding[d] = 0.05f;
        uint64_t id = chunks.put(c);
        engine.onChunkInserted(id);
        return id;
    }

    void removeChunk(uint64_t id) {
        chunks.remove(id);
        engine.onChunkRemoved(id);
    }

    std::set<std::pair<uint64_t, std::string>> chunkState() const {
        std::set<std::pair<uint64_t, std::string>> state;
        chunks.forEach([&](const ChunkNode& c) {
            state.insert({c.id, std::string(c.text)});
        });
        return state;
    }
};

// Full bidirectional exchange through the JSON wire format.
void exchange(Replica& a, Replica& b) {
    auto delta_a = SyncDelta::deserialize(a.engine.exportDelta(0).serialize());
    auto delta_b = SyncDelta::deserialize(b.engine.exportDelta(0).serialize());
    b.engine.applyDelta(delta_a);
    a.engine.applyDelta(delta_b);
}

} // namespace

TEST_CASE("Chunk ids are device-partitioned and never collide") {
    Replica a("device-A"), b("device-B");

    uint64_t id_a = a.addChunk("from A", 1, 0);
    uint64_t id_b = b.addChunk("from B", 1, 0);

    CHECK(id_a != id_b);
    CHECK((id_a >> 48) != 0);
    CHECK((id_b >> 48) != 0);
    CHECK((id_a >> 48) != (id_b >> 48));
}

TEST_CASE("Bidirectional exchange converges to identical chunk state") {
    Replica a("device-A"), b("device-B");

    a.addChunk("alpha", 1, 0);
    a.addChunk("beta", 1, 1);
    b.addChunk("gamma", 2, 0);

    exchange(a, b);
    exchange(a, b); // settle round

    CHECK(a.chunks.size() == 3);
    CHECK(b.chunks.size() == 3);
    CHECK(a.chunkState() == b.chunkState()); // ids AND text identical
}

TEST_CASE("Chunk deletes propagate across replicas") {
    Replica a("device-A"), b("device-B");

    uint64_t id = a.addChunk("doomed", 1, 0);
    a.addChunk("survivor", 1, 1);

    exchange(a, b);
    CHECK(b.chunks.size() == 2);

    a.removeChunk(id);
    exchange(a, b);

    CHECK(a.chunks.size() == 1);
    CHECK(b.chunks.size() == 1);
    ChunkNode dummy;
    CHECK_FALSE(b.chunks.get(id, dummy));
    CHECK(a.chunkState() == b.chunkState());
}

TEST_CASE("Delta re-application is idempotent") {
    Replica a("device-A"), b("device-B");

    a.addChunk("one", 1, 0);
    a.addChunk("two", 1, 1);

    auto json = a.engine.exportDelta(0).serialize();
    auto delta = SyncDelta::deserialize(json);

    auto r1 = b.engine.applyDelta(delta);
    CHECK(r1.chunks_applied == 2);

    auto r2 = b.engine.applyDelta(delta);
    CHECK(r2.chunks_applied == 0); // nothing new the second time
    CHECK(b.chunks.size() == 2);
}

TEST_CASE("Object LWW with interleaved updates converges") {
    Replica a("device-A"), b("device-B");

    // Same object id on both sides (objects share the id space by design;
    // conflicting writes to the same id resolve by LWW).
    nlohmann::json pa; pa["value"] = "from_A";
    uint64_t ida = a.objects.put("Shared", pa);
    a.engine.onRecordMutated(ida);

    nlohmann::json pb; pb["value"] = "from_B";
    uint64_t idb = b.objects.put("Shared", pb);
    b.engine.onRecordMutated(idb);
    // B mutates again → higher clock → B must win everywhere.
    nlohmann::json pb2; pb2["value"] = "from_B_v2"; pb2["id"] = idb;
    b.objects.put("Shared", pb2);
    b.engine.onRecordMutated(idb);

    exchange(a, b);
    exchange(a, b);

    nlohmann::json out_a, out_b;
    REQUIRE(a.objects.get(ida, out_a));
    REQUIRE(b.objects.get(idb, out_b));
    CHECK(out_a["value"].get<std::string>() == "from_B_v2");
    CHECK(out_b["value"].get<std::string>() == "from_B_v2");
}

TEST_CASE("Logical clock merges forward (Lamport)") {
    Replica a("device-A"), b("device-B");

    for (int i = 0; i < 5; i++) a.addChunk("c" + std::to_string(i), 1, i);
    CHECK(a.engine.getCurrentClock() == 5);
    CHECK(b.engine.getCurrentClock() == 0);

    auto delta = SyncDelta::deserialize(a.engine.exportDelta(0).serialize());
    b.engine.applyDelta(delta);

    // B's clock must move past A's so B's next local edit orders after.
    CHECK(b.engine.getCurrentClock() >= 5);
}

TEST_CASE("Corrupt delta JSON fails closed") {
    auto delta = SyncDelta::deserialize("{\"source_device_id\": 12, garbage");
    CHECK(delta.records.empty());
    CHECK(delta.chunks.empty());
    CHECK(delta.edges.empty());
}
