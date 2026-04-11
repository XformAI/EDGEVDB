#pragma once

#include "kg_extractor.hpp"
#include "log.hpp"

#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
#include <string>
#include <vector>
#include <queue>

namespace edgevdb {

class KnowledgeGraph {
public:
    KnowledgeGraph();
    explicit KnowledgeGraph(const std::string& file_path);

    void addChunkEntities(uint64_t chunk_id, const std::vector<Entity>& entities);
    std::vector<uint64_t> getChunksForEntity(const std::string& entity) const;
    std::unordered_set<std::string> getRelatedEntities(const std::string& entity, int hops = 1) const;
    bool removeChunk(uint64_t chunk_id);
    void clear();
    bool save();
    bool open();

private:
    std::unordered_map<std::string, std::vector<uint64_t>> entity_to_chunks_;
    std::unordered_map<std::string, std::unordered_set<std::string>> entity_cooccurrence_;
    std::string file_path_;
    mutable std::shared_mutex rw_mutex_;
};

} // namespace edgevdb
