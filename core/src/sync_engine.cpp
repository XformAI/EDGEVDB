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

    // Serialize chunks (simplified — just id and text)
    auto chunks_arr = nlohmann::json::array();
    for (const auto& chunk : chunks) {
        nlohmann::json c;
        c["id"] = chunk.id;
        c["text"] = std::string(chunk.text);
        c["doc_id"] = chunk.doc_id;
        c["page_number"] = chunk.page_number;
        c["insert_timestamp"] = chunk.insert_timestamp;
        chunks_arr.push_back(c);
    }
    j["chunks"] = chunks_arr;

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
                nlohmann::json r = j["records"][i];
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
                nlohmann::json e = j["edges"][i];
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
                nlohmann::json ch = j["chunks"][i];
                ChunkNode chunk;
                chunk.id = ch["id"].get<uint64_t>();
                std::string txt = ch["text"].get<std::string>();
                std::strncpy(chunk.text, txt.c_str(), MAX_TEXT_LEN - 1);
                chunk.text[MAX_TEXT_LEN - 1] = '\0';
                chunk.doc_id = ch["doc_id"].get<uint32_t>();
                chunk.page_number = ch["page_number"].get<uint32_t>();
                chunk.insert_timestamp = ch["insert_timestamp"].get<uint64_t>();
                delta.chunks.push_back(chunk);
            }
        }
    } catch (...) {
        EVDB_LOG_ERROR("SyncEngine: Failed to deserialize delta");
    }
    return delta;
}

// ── SyncEngine ────────────────────────────────────────────

SyncEngine::SyncEngine(const std::string& device_id,
                       ObjectStore* obj_store,
                       RelationIndex* rel_index,
                       ChunkStore* chunk_store)
    : obj_store_(obj_store), rel_index_(rel_index), chunk_store_(chunk_store) {
    clock_.device_id = device_id;
    clock_.logical_clock = 0;
}

SyncDelta SyncEngine::exportDelta(uint64_t since_clock) const {
    SyncDelta delta;
    delta.source_device_id = clock_.device_id;
    delta.from_clock = since_clock;
    delta.to_clock = clock_.logical_clock;

    // Collect all ObjectRecords modified after since_clock
    if (obj_store_) {
        obj_store_->forEach([&](const ObjectRecord& rec) {
            if (rec.sync_vector_clock > since_clock) {
                delta.records.push_back(rec);
            }
        });
    }

    // Collect relation edges
    if (rel_index_) {
        rel_index_->forEach([&](const RelationEdge& edge) {
            delta.edges.push_back(edge);
        });
    }

    // Collect chunks inserted after since_clock
    if (chunk_store_) {
        chunk_store_->forEach([&](const ChunkNode& chunk) {
            if (chunk.insert_timestamp > since_clock) {
                delta.chunks.push_back(chunk);
            }
        });
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

    // Apply edges
    for (const auto& edge : delta.edges) {
        if (rel_index_) {
            rel_index_->addRelation(std::string(edge.relation_name), edge.from_id, edge.to_id);
            result.edges_applied++;
        }
    }

    // Apply chunks (immutable — only insert if not present)
    for (const auto& chunk : delta.chunks) {
        if (chunk_store_) {
            ChunkNode existing;
            if (!chunk_store_->get(chunk.id, existing)) {
                ChunkNode new_chunk = chunk;
                chunk_store_->put(new_chunk);
                result.chunks_applied++;
            }
        }
    }

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
