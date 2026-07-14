#pragma once

#include "schema.hpp"
#include "log.hpp"
#include "crc32.hpp"
#include "persist.hpp"

#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <functional>

namespace edgevdb {

class RelationIndex {
public:
    RelationIndex();

    void addRelation(const std::string& name, uint64_t from_id, uint64_t to_id);
    std::vector<uint64_t> getTargets(const std::string& name, uint64_t from_id) const;
    std::vector<uint64_t> getSources(const std::string& name, uint64_t to_id) const;
    bool removeRelation(const std::string& name, uint64_t from_id, uint64_t to_id);
    void removeAllRelationsForRecord(uint64_t record_id);
    bool save(const std::string& path) const;
    bool open(const std::string& path);
    void clear();

    // For sync
    void forEach(std::function<void(const RelationEdge&)> fn) const;

private:
    std::unordered_map<std::string, std::vector<std::pair<uint64_t, uint64_t>>> relation_edges_;
    std::unordered_map<std::string, std::unordered_map<uint64_t, std::vector<uint64_t>>> from_index_;
    std::unordered_map<std::string, std::unordered_map<uint64_t, std::vector<uint64_t>>> to_index_;
    mutable std::shared_mutex rw_mutex_;

    void clearUnlocked();
};

// ── Implementation ────────────────────────────────────────

inline RelationIndex::RelationIndex() {}

inline void RelationIndex::addRelation(const std::string& name, uint64_t from_id, uint64_t to_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    // Check for duplicate
    auto& edges = from_index_[name][from_id];
    if (std::find(edges.begin(), edges.end(), to_id) != edges.end()) return;

    relation_edges_[name].push_back({from_id, to_id});
    from_index_[name][from_id].push_back(to_id);
    to_index_[name][to_id].push_back(from_id);
}

inline std::vector<uint64_t> RelationIndex::getTargets(const std::string& name, uint64_t from_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = from_index_.find(name);
    if (it == from_index_.end()) return {};
    auto it2 = it->second.find(from_id);
    if (it2 == it->second.end()) return {};
    return it2->second;
}

inline std::vector<uint64_t> RelationIndex::getSources(const std::string& name, uint64_t to_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = to_index_.find(name);
    if (it == to_index_.end()) return {};
    auto it2 = it->second.find(to_id);
    if (it2 == it->second.end()) return {};
    return it2->second;
}

inline bool RelationIndex::removeRelation(const std::string& name, uint64_t from_id, uint64_t to_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    auto& edges = relation_edges_[name];
    edges.erase(std::remove(edges.begin(), edges.end(), std::make_pair(from_id, to_id)), edges.end());

    auto& targets = from_index_[name][from_id];
    targets.erase(std::remove(targets.begin(), targets.end(), to_id), targets.end());

    auto& sources = to_index_[name][to_id];
    sources.erase(std::remove(sources.begin(), sources.end(), from_id), sources.end());

    return true;
}

inline void RelationIndex::removeAllRelationsForRecord(uint64_t record_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    for (auto& [name, edges] : relation_edges_) {
        edges.erase(std::remove_if(edges.begin(), edges.end(),
            [record_id](const std::pair<uint64_t, uint64_t>& e) {
                return e.first == record_id || e.second == record_id;
            }), edges.end());
    }

    for (auto& [name, from_map] : from_index_) {
        from_map.erase(record_id);
        for (auto& [from_id, targets] : from_map) {
            targets.erase(std::remove(targets.begin(), targets.end(), record_id), targets.end());
        }
    }

    for (auto& [name, to_map] : to_index_) {
        to_map.erase(record_id);
        for (auto& [to_id, sources] : to_map) {
            sources.erase(std::remove(sources.begin(), sources.end(), record_id), sources.end());
        }
    }
}

inline void RelationIndex::forEach(std::function<void(const RelationEdge&)> fn) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    for (const auto& [name, edges] : relation_edges_) {
        for (const auto& [from_id, to_id] : edges) {
            RelationEdge edge;
            std::strncpy(edge.relation_name, name.c_str(), 63);
            edge.relation_name[63] = '\0';
            edge.from_id = from_id;
            edge.to_id = to_id;
            fn(edge);
        }
    }
}

inline bool RelationIndex::save(const std::string& path) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    return persist::atomicSave(path, [this](std::ofstream& file) {
        const char magic[8] = {'E','V','D','B','R','E','L','\0'};
        file.write(magic, 8);

        uint32_t version = 2; // v2: CRC enforced on load
        file.write(reinterpret_cast<const char*>(&version), 4);

        // Count total edges
        uint32_t total = 0;
        for (const auto& [name, edges] : relation_edges_) total += static_cast<uint32_t>(edges.size());
        file.write(reinterpret_cast<const char*>(&total), 4);

        uint32_t crc_state = CRC32_INIT;
        for (const auto& [name, edges] : relation_edges_) {
            for (const auto& [from_id, to_id] : edges) {
                RelationEdge edge{};
                std::strncpy(edge.relation_name, name.c_str(), 63);
                edge.from_id = from_id;
                edge.to_id = to_id;
                file.write(reinterpret_cast<const char*>(&edge), sizeof(RelationEdge));
                crc_state = crc32Update(crc_state, &edge, sizeof(RelationEdge));
            }
        }

        uint32_t crc = crc32Finalize(crc_state);
        file.write(reinterpret_cast<const char*>(&crc), 4);
        return file.good();
    });
}

inline bool RelationIndex::open(const std::string& path) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return true; // no file = empty

    const uint64_t total_size = persist::fileSize(file);
    constexpr uint64_t HEADER_SIZE = 8 + 4 + 4;

    char magic[8];
    file.read(magic, 8);
    const char expected[8] = {'E','V','D','B','R','E','L','\0'};
    if (std::memcmp(magic, expected, 8) != 0) {
        EVDB_LOG_ERROR("RelationIndex: Invalid magic");
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != 2) {
        EVDB_LOG_ERROR("RelationIndex: Unsupported version %u (expected 2)", version);
        return false;
    }

    uint32_t count;
    file.read(reinterpret_cast<char*>(&count), 4);
    if (total_size < HEADER_SIZE + 4 ||
        !persist::boundedCount(count, sizeof(RelationEdge), total_size - HEADER_SIZE - 4)) {
        EVDB_LOG_ERROR("RelationIndex: Corrupt count %u", count);
        return false;
    }

    clearUnlocked();

    uint32_t crc_state = CRC32_INIT;
    for (uint32_t i = 0; i < count; i++) {
        RelationEdge edge;
        file.read(reinterpret_cast<char*>(&edge), sizeof(RelationEdge));
        if (!file.good()) { clearUnlocked(); return false; }
        crc_state = crc32Update(crc_state, &edge, sizeof(RelationEdge));

        edge.relation_name[63] = '\0';
        std::string name(edge.relation_name);
        relation_edges_[name].push_back({edge.from_id, edge.to_id});
        from_index_[name][edge.from_id].push_back(edge.to_id);
        to_index_[name][edge.to_id].push_back(edge.from_id);
    }

    uint32_t stored_crc;
    file.read(reinterpret_cast<char*>(&stored_crc), 4);
    if (!file.good() || crc32Finalize(crc_state) != stored_crc) {
        EVDB_LOG_ERROR("RelationIndex: CRC32 mismatch — refusing corrupt data");
        clearUnlocked();
        return false;
    }

    return true;
}

inline void RelationIndex::clearUnlocked() {
    relation_edges_.clear();
    from_index_.clear();
    to_index_.clear();
}

inline void RelationIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    clearUnlocked();
}

} // namespace edgevdb
