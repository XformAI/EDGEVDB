#include "sync_engine.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>

namespace edgevdb {

// ── SyncDelta serialization ───────────────────────────────

std::string SyncDelta::serialize() const {
    nlohmann::json j;
    j["source_device_id"] = source_device_id;
    j["from_clock"] = from_clock;
    j["to_clock"] = to_clock;

    // Serialize records
    auto records_arr = nlohmann::json::array();
    for (const auto& rec : records) {
        nlohmann::json r;
        r["id"] = rec.id;
        r["type_name"] = std::string(rec.type_name);
        r["data"] = std::string(reinterpret_cast<const char*>(rec.data));
        r["created_at"] = rec.created_at;
        r["updated_at"] = rec.updated_at;
        r["sync_vector_clock"] = rec.sync_vector_clock;
        r["is_deleted"] = rec.is_deleted;
        records_arr.push_back(r);
    }
    j["records"] = records_arr;

    // Serialize edges
    auto edges_arr = nlohmann::json::array();
    for (const auto& edge : edges) {
        nlohmann::json e;
        e["from_id"] = edge.from_id;
        e["to_id"] = edge.to_id;
        e["relation_name"] = std::string(edge.relation_name);
        e["created_at"] = edge.created_at;
        edges_arr.push_back(e);
    }
    j["edges"] = edges_arr;

    // Serialize chunks. Full fidelity: text, placement, embedding and the
    // logical sync clock, so a replica can index the chunk identically.
    auto chunks_arr = nlohmann::json::array();
    for (const auto& chunk : chunks) {
        nlohmann::json c;
        c["id"] = chunk.id;
        c["text"] = std::string(chunk.text);
        c["doc_id"] = chunk.doc_id;
        c["page_number"] = chunk.page_number;
        c["insert_timestamp"] = chunk.insert_timestamp;
        c["sync_clock"] = chunkSyncClock(chunk);
        auto emb = nlohmann::json::array();
        for (size_t d = 0; d < EMBEDDING_DIM; d++) emb.push_back(chunk.embedding[d]);
        c["embedding"] = emb;
        chunks_arr.push_back(c);
    }
    j["chunks"] = chunks_arr;

    // Chunk delete tombstones so removals propagate across replicas.
    auto dels_arr = nlohmann::json::array();
    for (const auto& t : chunk_deletes) {
        nlohmann::json d;
        d["id"] = t.chunk_id;
        d["clock"] = t.clock;
        dels_arr.push_back(d);
    }
    j["chunk_deletes"] = dels_arr;

    return j.dump();
}

SyncDelta SyncDelta::deserialize(const std::string& json_str) {
    SyncDelta delta;
    try {
        auto j = nlohmann::json::parse(json_str);
        delta.source_device_id = j["source_device_id"].get<std::string>();
        delta.from_clock = j["from_clock"].get<uint64_t>();
        delta.to_clock = j["to_clock"].get<uint64_t>();

        if (j.contains("records") && j["records"].is_array()) {
            for (size_t i = 0; i < j["records"].size(); i++) {
                const nlohmann::json& r = j["records"][i];
                ObjectRecord rec;
                rec.id = r["id"].get<uint64_t>();
                std::string tn = r["type_name"].get<std::string>();
                std::strncpy(rec.type_name, tn.c_str(), 63);
                rec.type_name[63] = '\0';
                std::string data_str = r["data"].get<std::string>();
                std::memset(rec.data, 0, MAX_OBJECT_DATA);
                std::memcpy(rec.data, data_str.c_str(),
                            std::min(data_str.size(), static_cast<size_t>(MAX_OBJECT_DATA - 1)));
                rec.created_at = r["created_at"].get<uint64_t>();
                rec.updated_at = r["updated_at"].get<uint64_t>();
                rec.sync_vector_clock = r["sync_vector_clock"].get<uint64_t>();
                rec.is_deleted = r["is_deleted"].get<bool>();
                delta.records.push_back(rec);
            }
        }

        if (j.contains("edges") && j["edges"].is_array()) {
            for (size_t i = 0; i < j["edges"].size(); i++) {
                const nlohmann::json& e = j["edges"][i];
                RelationEdge edge;
                edge.from_id = e["from_id"].get<uint64_t>();
                edge.to_id = e["to_id"].get<uint64_t>();
                std::string rn = e["relation_name"].get<std::string>();
                std::strncpy(edge.relation_name, rn.c_str(), 63);
                edge.relation_name[63] = '\0';
                edge.created_at = e["created_at"].get<uint64_t>();
                delta.edges.push_back(edge);
            }
        }

        if (j.contains("chunks") && j["chunks"].is_array()) {
            for (size_t i = 0; i < j["chunks"].size(); i++) {
                const nlohmann::json& ch = j["chunks"][i];
                ChunkNode chunk;
                chunk.id = ch["id"].get<uint64_t>();
                std::string txt = ch["text"].get<std::string>();
                std::strncpy(chunk.text, txt.c_str(), MAX_TEXT_LEN - 1);
                chunk.text[MAX_TEXT_LEN - 1] = '\0';
                chunk.doc_id = ch["doc_id"].get<uint32_t>();
                chunk.page_number = ch["page_number"].get<uint32_t>();
                chunk.insert_timestamp = ch["insert_timestamp"].get<uint64_t>();
                if (ch.contains("sync_clock")) {
                    setChunkSyncClock(chunk, ch["sync_clock"].get<uint64_t>());
                }
                if (ch.contains("embedding") && ch["embedding"].is_array() &&
                    ch["embedding"].size() == EMBEDDING_DIM) {
                    for (size_t d = 0; d < EMBEDDING_DIM; d++) {
                        chunk.embedding[d] = ch["embedding"][d].get<float>();
                    }
                }
                delta.chunks.push_back(chunk);
            }
        }

        if (j.contains("chunk_deletes") && j["chunk_deletes"].is_array()) {
            for (size_t i = 0; i < j["chunk_deletes"].size(); i++) {
                const nlohmann::json& d = j["chunk_deletes"][i];
                ChunkTombstone t;
                t.chunk_id = d["id"].get<uint64_t>();
                t.clock = d["clock"].get<uint64_t>();
                delta.chunk_deletes.push_back(t);
            }
        }
    } catch (const std::exception& e) {
        EVDB_LOG_ERROR("SyncEngine: Failed to deserialize delta: %s", e.what());
        return SyncDelta{}; // fail closed: empty delta, never partial data
    }
    return delta;
}

// ── SyncEngine ────────────────────────────────────────────

namespace {
// FNV-1a hash of the device id, folded to 16 bits, used to partition the
// chunk id space: high 16 bits = device hash, low 48 bits = local counter.
uint16_t deviceHash16(const std::string& device_id) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : device_id) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    uint16_t folded = static_cast<uint16_t>((h ^ (h >> 16) ^ (h >> 32) ^ (h >> 48)) & 0xFFFF);
    return folded == 0 ? 1 : folded; // 0 is reserved for un-partitioned stores
}
} // namespace

SyncEngine::SyncEngine(const std::string& device_id,
                       ObjectStore* obj_store,
                       RelationIndex* rel_index,
                       ChunkStore* chunk_store)
    : obj_store_(obj_store), rel_index_(rel_index), chunk_store_(chunk_store) {
    clock_.device_id = device_id;
    clock_.logical_clock = 0;
    device_id_base_ = static_cast<uint64_t>(deviceHash16(device_id)) << 48;
    if (chunk_store_) {
        chunk_store_->setIdBase(device_id_base_);
    }
}

SyncDelta SyncEngine::exportDelta(uint64_t since_clock) const {
    SyncDelta delta;
    delta.source_device_id = clock_.device_id;
    delta.from_clock = since_clock;
    delta.to_clock = clock_.logical_clock;

    // Collect all ObjectRecords modified after since_clock (soft-deleted
    // records are included: they are the delete tombstones for objects).
    if (obj_store_) {
        obj_store_->forEach([&](const ObjectRecord& rec) {
            if (rec.sync_vector_clock > since_clock) {
                delta.records.push_back(rec);
            }
        });
    }

    // Collect relation edges (idempotent on apply, so full send is safe).
    if (rel_index_) {
        rel_index_->forEach([&](const RelationEdge& edge) {
            delta.edges.push_back(edge);
        });
    }

    // Collect chunks stamped with a logical clock after since_clock.
    // Comparing logical clocks (not wall time) keeps delta filtering exact.
    if (chunk_store_) {
        chunk_store_->forEach([&](const ChunkNode& chunk) {
            if (chunkSyncClock(chunk) > since_clock) {
                delta.chunks.push_back(chunk);
            }
        });
    }

    // Chunk delete tombstones after since_clock.
    for (const auto& t : chunk_tombstones_) {
        if (t.clock > since_clock) {
            delta.chunk_deletes.push_back(t);
        }
    }

    return delta;
}

MergeResult SyncEngine::applyDelta(const SyncDelta& delta) {
    MergeResult result;

    // Apply records
    for (const auto& remote_rec : delta.records) {
        if (!obj_store_) continue;

        ObjectRecord local_rec;
        bool local_exists = obj_store_->getRecord(remote_rec.id, local_rec);

        if (!local_exists) {
            // Insert new record
            obj_store_->putRecord(remote_rec);
            result.records_applied++;
        } else {
            // Compare vector clocks for LWW
            if (remote_rec.sync_vector_clock > local_rec.sync_vector_clock) {
                // Remote is newer → apply
                obj_store_->putRecord(remote_rec);
                result.records_applied++;
                result.conflicts_lww_resolved++;
            } else if (remote_rec.sync_vector_clock == local_rec.sync_vector_clock) {
                // Tie-break by device ID (lexicographic)
                if (delta.source_device_id > clock_.device_id) {
                    obj_store_->putRecord(remote_rec);
                    result.records_applied++;
                    result.conflicts_lww_resolved++;
                } else {
                    result.records_skipped_older++;
                }
            } else {
                result.records_skipped_older++;
            }
        }
    }

    // Apply edges (addRelation is idempotent: duplicates are ignored)
    for (const auto& edge : delta.edges) {
        if (rel_index_) {
            rel_index_->addRelation(std::string(edge.relation_name), edge.from_id, edge.to_id);
            result.edges_applied++;
        }
    }

    // Apply chunks. Ids are device-partitioned so preserving the remote id
    // via putWithId keeps replicas identical (chunks are immutable).
    // A chunk already deleted locally (tombstone with a newer clock) must
    // NOT be resurrected by a delta exported before the delete propagated.
    auto tombstoned = [this](const ChunkNode& chunk) {
        for (const auto& t : chunk_tombstones_) {
            if (t.chunk_id == chunk.id && t.clock >= chunkSyncClock(chunk)) return true;
        }
        return false;
    };
    for (const auto& chunk : delta.chunks) {
        if (chunk_store_ && !tombstoned(chunk)) {
            ChunkNode existing;
            if (!chunk_store_->get(chunk.id, existing)) {
                chunk_store_->putWithId(chunk.id, chunk);
                result.chunks_applied++;
                if (on_chunk_applied_) on_chunk_applied_(chunk);
            }
        }
    }

    // Apply chunk deletions and remember them so they propagate onward.
    for (const auto& t : delta.chunk_deletes) {
        if (chunk_store_ && chunk_store_->remove(t.chunk_id)) {
            result.chunks_deleted++;
            if (on_chunk_deleted_) on_chunk_deleted_(t.chunk_id);
        }
        bool known = std::any_of(chunk_tombstones_.begin(), chunk_tombstones_.end(),
                                 [&](const ChunkTombstone& x) { return x.chunk_id == t.chunk_id; });
        if (!known) chunk_tombstones_.push_back(t);
    }

    // Lamport merge so subsequent local mutations order after the remote's.
    clock_.observe(delta.to_clock);

    return result;
}

uint64_t SyncEngine::getCurrentClock() const {
    return clock_.logical_clock;
}

void SyncEngine::onRecordMutated(uint64_t record_id) {
    clock_.tick();
    // Update the record's sync_vector_clock
    if (obj_store_) {
        ObjectRecord rec;
        if (obj_store_->getRecord(record_id, rec)) {
            rec.sync_vector_clock = clock_.logical_clock;
            obj_store_->putRecord(rec);
        }
    }
}

void SyncEngine::onChunkInserted(uint64_t chunk_id) {
    clock_.tick();
    if (chunk_store_) {
        ChunkNode chunk;
        if (chunk_store_->get(chunk_id, chunk)) {
            setChunkSyncClock(chunk, clock_.logical_clock);
            chunk_store_->putWithId(chunk_id, chunk);
        }
    }
}

void SyncEngine::onChunkRemoved(uint64_t chunk_id) {
    clock_.tick();
    chunk_tombstones_.push_back({chunk_id, clock_.logical_clock});
}

bool SyncEngine::exportToFile(const std::string& path, uint64_t since_clock) const {
    SyncDelta delta = exportDelta(since_clock);
    std::string json = delta.serialize();

    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << json;
    return file.good();
}

bool SyncEngine::importFromFile(const std::string& path, MergeResult& result) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string json((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());

    SyncDelta delta = SyncDelta::deserialize(json);
    result = applyDelta(delta);
    return true;
}

} // namespace edgevdb
