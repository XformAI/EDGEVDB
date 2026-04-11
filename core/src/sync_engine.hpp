#pragma once

#include "schema.hpp"
#include "sync_protocol.hpp"
#include "object_store.hpp"
#include "relation_index.hpp"
#include "chunk_store.hpp"
#include "log.hpp"
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <cstdint>

namespace edgevdb {

struct DeviceClock {
    std::string device_id;
    uint64_t logical_clock = 0;

    uint64_t tick() { return ++logical_clock; }
    bool happensAfter(uint64_t a, uint64_t b) const { return a > b; }
};

struct SyncDelta {
    std::string source_device_id;
    uint64_t from_clock = 0;
    uint64_t to_clock = 0;
    std::vector<ObjectRecord> records;
    std::vector<RelationEdge> edges;
    std::vector<ChunkNode> chunks;

    std::string serialize() const;
    static SyncDelta deserialize(const std::string& json_str);
};

struct MergeResult {
    int records_applied = 0;
    int records_skipped_older = 0;
    int conflicts_lww_resolved = 0;
    int chunks_applied = 0;
    int edges_applied = 0;
};

class SyncEngine {
public:
    SyncEngine(const std::string& device_id,
               ObjectStore* obj_store,
               RelationIndex* rel_index,
               ChunkStore* chunk_store);

    SyncDelta exportDelta(uint64_t since_clock) const;
    MergeResult applyDelta(const SyncDelta& delta);

    uint64_t getCurrentClock() const;
    void onRecordMutated(uint64_t record_id);

    bool exportToFile(const std::string& path, uint64_t since_clock) const;
    bool importFromFile(const std::string& path, MergeResult& result);

private:
    DeviceClock clock_;
    ObjectStore* obj_store_;
    RelationIndex* rel_index_;
    ChunkStore* chunk_store_;
};

} // namespace edgevdb
