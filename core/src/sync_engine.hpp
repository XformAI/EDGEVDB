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
#include <functional>
#include <cstdint>
#include <cstring>

namespace edgevdb {

struct DeviceClock {
    std::string device_id;
    uint64_t logical_clock = 0;

    uint64_t tick() { return ++logical_clock; }
    // Lamport merge: fold in a remote clock so later local events order
    // after everything we've already observed.
    void observe(uint64_t remote_clock) {
        if (remote_clock > logical_clock) logical_clock = remote_clock;
    }
    bool happensAfter(uint64_t a, uint64_t b) const { return a > b; }
};

// Chunk sync clocks live in ChunkNode.reserved[0..7] (spare bytes) so the
// on-disk chunk format does not change size.
inline uint64_t chunkSyncClock(const ChunkNode& chunk) {
    uint64_t clock;
    std::memcpy(&clock, chunk.reserved, 8);
    return clock;
}
inline void setChunkSyncClock(ChunkNode& chunk, uint64_t clock) {
    std::memcpy(chunk.reserved, &clock, 8);
}

struct ChunkTombstone {
    uint64_t chunk_id = 0;
    uint64_t clock = 0;
};

struct SyncDelta {
    std::string source_device_id;
    uint64_t from_clock = 0;
    uint64_t to_clock = 0;
    std::vector<ObjectRecord> records;
    std::vector<RelationEdge> edges;
    std::vector<ChunkNode> chunks;
    std::vector<ChunkTombstone> chunk_deletes;

    std::string serialize() const;
    static SyncDelta deserialize(const std::string& json_str);
};

struct MergeResult {
    int records_applied = 0;
    int records_skipped_older = 0;
    int conflicts_lww_resolved = 0;
    int chunks_applied = 0;
    int chunks_deleted = 0;
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

    // Chunk lifecycle notifications. Chunk ids are partitioned per device
    // (high 16 bits = device hash) so ids created on different replicas
    // never collide and survive sync unchanged.
    void onChunkInserted(uint64_t chunk_id);
    void onChunkRemoved(uint64_t chunk_id);

    // Hooks the owner uses to keep secondary indexes (HNSW, page index,
    // knowledge graph) in step with chunks arriving/leaving via sync.
    void setChunkAppliedCallback(std::function<void(const ChunkNode&)> cb) {
        on_chunk_applied_ = std::move(cb);
    }
    void setChunkDeletedCallback(std::function<void(uint64_t)> cb) {
        on_chunk_deleted_ = std::move(cb);
    }

    uint64_t deviceIdBase() const { return device_id_base_; }

    bool exportToFile(const std::string& path, uint64_t since_clock) const;
    bool importFromFile(const std::string& path, MergeResult& result);

private:
    DeviceClock clock_;
    ObjectStore* obj_store_;
    RelationIndex* rel_index_;
    ChunkStore* chunk_store_;
    uint64_t device_id_base_ = 0;
    std::vector<ChunkTombstone> chunk_tombstones_;
    std::function<void(const ChunkNode&)> on_chunk_applied_;
    std::function<void(uint64_t)> on_chunk_deleted_;
};

} // namespace edgevdb
