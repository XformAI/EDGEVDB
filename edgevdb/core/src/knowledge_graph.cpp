#include "knowledge_graph.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>

namespace edgevdb {

KnowledgeGraph::KnowledgeGraph() {}

KnowledgeGraph::KnowledgeGraph(const std::string& file_path) : file_path_(file_path) {}

void KnowledgeGraph::addChunkEntities(uint64_t chunk_id, const std::vector<Entity>& entities) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    for (const auto& entity : entities) {
        entity_to_chunks_[entity.text].push_back(chunk_id);
    }

    // Add co-occurrence edges for all pairs in same chunk
    for (size_t i = 0; i < entities.size(); i++) {
        for (size_t j = i + 1; j < entities.size(); j++) {
            entity_cooccurrence_[entities[i].text].insert(entities[j].text);
            entity_cooccurrence_[entities[j].text].insert(entities[i].text);
        }
    }
}

std::vector<uint64_t> KnowledgeGraph::getChunksForEntity(const std::string& entity) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = entity_to_chunks_.find(entity);
    if (it == entity_to_chunks_.end()) return {};
    return it->second;
}

std::unordered_set<std::string> KnowledgeGraph::getRelatedEntities(const std::string& entity, int hops) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    std::unordered_set<std::string> visited;
    std::queue<std::pair<std::string, int>> queue;
    queue.push({entity, 0});
    visited.insert(entity);

    while (!queue.empty()) {
        auto [current, depth] = queue.front();
        queue.pop();

        if (depth >= hops) continue;

        auto it = entity_cooccurrence_.find(current);
        if (it != entity_cooccurrence_.end()) {
            for (const auto& neighbor : it->second) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.push({neighbor, depth + 1});
                }
            }
        }
    }

    visited.erase(entity); // remove self
    return visited;
}

bool KnowledgeGraph::removeChunk(uint64_t chunk_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    for (auto& [entity, chunks] : entity_to_chunks_) {
        chunks.erase(std::remove(chunks.begin(), chunks.end(), chunk_id), chunks.end());
    }
    return true;
}

void KnowledgeGraph::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    entity_to_chunks_.clear();
    entity_cooccurrence_.clear();
}

bool KnowledgeGraph::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    if (file_path_.empty()) return false;

    std::ofstream file(file_path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return false;

    const char magic[8] = {'E','V','D','B','K','G','\0','\0'};
    file.write(magic, 8);

    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), 4);

    // Write entity_to_chunks
    uint32_t entity_count = static_cast<uint32_t>(entity_to_chunks_.size());
    file.write(reinterpret_cast<const char*>(&entity_count), 4);

    for (const auto& [entity, chunks] : entity_to_chunks_) {
        uint32_t name_len = static_cast<uint32_t>(entity.size());
        file.write(reinterpret_cast<const char*>(&name_len), 4);
        file.write(entity.data(), name_len);

        uint32_t chunk_count = static_cast<uint32_t>(chunks.size());
        file.write(reinterpret_cast<const char*>(&chunk_count), 4);
        for (uint64_t cid : chunks) {
            file.write(reinterpret_cast<const char*>(&cid), 8);
        }
    }

    // Write co-occurrence edges
    uint32_t cooc_count = static_cast<uint32_t>(entity_cooccurrence_.size());
    file.write(reinterpret_cast<const char*>(&cooc_count), 4);

    for (const auto& [entity, neighbors] : entity_cooccurrence_) {
        uint32_t name_len = static_cast<uint32_t>(entity.size());
        file.write(reinterpret_cast<const char*>(&name_len), 4);
        file.write(entity.data(), name_len);

        uint32_t nb_count = static_cast<uint32_t>(neighbors.size());
        file.write(reinterpret_cast<const char*>(&nb_count), 4);
        for (const auto& nb : neighbors) {
            uint32_t nb_len = static_cast<uint32_t>(nb.size());
            file.write(reinterpret_cast<const char*>(&nb_len), 4);
            file.write(nb.data(), nb_len);
        }
    }

    return file.good();
}

bool KnowledgeGraph::open() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (file_path_.empty()) return true;

    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) return true; // no file = empty graph

    char magic[8];
    file.read(magic, 8);
    const char expected[8] = {'E','V','D','B','K','G','\0','\0'};
    if (std::memcmp(magic, expected, 8) != 0) return false;

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);

    // Read entity_to_chunks
    uint32_t entity_count;
    file.read(reinterpret_cast<char*>(&entity_count), 4);
    entity_to_chunks_.clear();

    for (uint32_t i = 0; i < entity_count; i++) {
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        uint32_t chunk_count;
        file.read(reinterpret_cast<char*>(&chunk_count), 4);
        std::vector<uint64_t> chunks(chunk_count);
        for (uint32_t j = 0; j < chunk_count; j++) {
            file.read(reinterpret_cast<char*>(&chunks[j]), 8);
        }
        entity_to_chunks_[name] = chunks;
    }

    // Read co-occurrence
    uint32_t cooc_count;
    file.read(reinterpret_cast<char*>(&cooc_count), 4);
    entity_cooccurrence_.clear();

    for (uint32_t i = 0; i < cooc_count; i++) {
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        uint32_t nb_count;
        file.read(reinterpret_cast<char*>(&nb_count), 4);
        for (uint32_t j = 0; j < nb_count; j++) {
            uint32_t nb_len;
            file.read(reinterpret_cast<char*>(&nb_len), 4);
            std::string nb(nb_len, '\0');
            file.read(&nb[0], nb_len);
            entity_cooccurrence_[name].insert(nb);
        }
    }

    EVDB_LOG_INFO("KnowledgeGraph: Loaded %u entities", entity_count);
    return true;
}

} // namespace edgevdb
