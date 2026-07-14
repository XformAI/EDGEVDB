#include "knowledge_graph.hpp"
#include "crc32.hpp"
#include "persist.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <mutex>

namespace edgevdb {

namespace {
// Sanity cap for any serialized string (entity names are short in practice).
constexpr uint32_t MAX_STRING_LEN = 1u << 20; // 1 MB
constexpr uint32_t KG_VERSION = 2;            // v2: CRC enforced, bounded reads
} // namespace

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

    // Remove the chunk from every entity's chunk list, remembering which
    // entities are left with no chunks at all.
    std::vector<std::string> orphaned;
    for (auto& [entity, chunks] : entity_to_chunks_) {
        auto before = chunks.size();
        chunks.erase(std::remove(chunks.begin(), chunks.end(), chunk_id), chunks.end());
        if (before != chunks.size() && chunks.empty()) {
            orphaned.push_back(entity);
        }
    }

    // Orphaned entities no longer exist in any chunk: drop them from the
    // entity map and from the co-occurrence graph (both directions) so the
    // graph doesn't accumulate stale edges.
    for (const auto& entity : orphaned) {
        entity_to_chunks_.erase(entity);
        auto it = entity_cooccurrence_.find(entity);
        if (it != entity_cooccurrence_.end()) {
            for (const auto& neighbor : it->second) {
                auto nb_it = entity_cooccurrence_.find(neighbor);
                if (nb_it != entity_cooccurrence_.end()) {
                    nb_it->second.erase(entity);
                    if (nb_it->second.empty()) entity_cooccurrence_.erase(nb_it);
                }
            }
            entity_cooccurrence_.erase(entity);
        }
    }
    return true;
}

void KnowledgeGraph::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    entity_to_chunks_.clear();
    entity_cooccurrence_.clear();
}

namespace {

void crcWrite(std::ofstream& file, uint32_t& crc_state, const void* data, size_t len) {
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(len));
    crc_state = crc32Update(crc_state, data, len);
}

bool crcReadString(std::ifstream& file, uint32_t& crc_state, uint64_t remaining,
                   std::string& out) {
    uint32_t len;
    file.read(reinterpret_cast<char*>(&len), 4);
    if (!file.good()) return false;
    crc_state = crc32Update(crc_state, &len, 4);
    if (len > MAX_STRING_LEN || len > remaining) return false;
    out.assign(len, '\0');
    if (len > 0) {
        file.read(&out[0], len);
        if (!file.good()) return false;
        crc_state = crc32Update(crc_state, out.data(), len);
    }
    return true;
}

} // namespace

bool KnowledgeGraph::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    if (file_path_.empty()) return false;

    return persist::atomicSave(file_path_, [this](std::ofstream& file) {
        const char magic[8] = {'E','V','D','B','K','G','\0','\0'};
        file.write(magic, 8);
        file.write(reinterpret_cast<const char*>(&KG_VERSION), 4);

        uint32_t crc_state = CRC32_INIT;

        // Write entity_to_chunks
        uint32_t entity_count = static_cast<uint32_t>(entity_to_chunks_.size());
        crcWrite(file, crc_state, &entity_count, 4);

        for (const auto& [entity, chunks] : entity_to_chunks_) {
            uint32_t name_len = static_cast<uint32_t>(entity.size());
            crcWrite(file, crc_state, &name_len, 4);
            crcWrite(file, crc_state, entity.data(), name_len);

            uint32_t chunk_count = static_cast<uint32_t>(chunks.size());
            crcWrite(file, crc_state, &chunk_count, 4);
            for (uint64_t cid : chunks) {
                crcWrite(file, crc_state, &cid, 8);
            }
        }

        // Write co-occurrence edges
        uint32_t cooc_count = static_cast<uint32_t>(entity_cooccurrence_.size());
        crcWrite(file, crc_state, &cooc_count, 4);

        for (const auto& [entity, neighbors] : entity_cooccurrence_) {
            uint32_t name_len = static_cast<uint32_t>(entity.size());
            crcWrite(file, crc_state, &name_len, 4);
            crcWrite(file, crc_state, entity.data(), name_len);

            uint32_t nb_count = static_cast<uint32_t>(neighbors.size());
            crcWrite(file, crc_state, &nb_count, 4);
            for (const auto& nb : neighbors) {
                uint32_t nb_len = static_cast<uint32_t>(nb.size());
                crcWrite(file, crc_state, &nb_len, 4);
                crcWrite(file, crc_state, nb.data(), nb_len);
            }
        }

        uint32_t crc = crc32Finalize(crc_state);
        file.write(reinterpret_cast<const char*>(&crc), 4);
        return file.good();
    });
}

bool KnowledgeGraph::open() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (file_path_.empty()) return true;

    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) return true; // no file = empty graph

    const uint64_t total_size = persist::fileSize(file);

    char magic[8];
    file.read(magic, 8);
    const char expected[8] = {'E','V','D','B','K','G','\0','\0'};
    if (std::memcmp(magic, expected, 8) != 0) {
        EVDB_LOG_ERROR("KnowledgeGraph: Invalid magic");
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != KG_VERSION) {
        EVDB_LOG_ERROR("KnowledgeGraph: Unsupported version %u (expected %u) — rebuild the graph",
                       version, KG_VERSION);
        return false;
    }

    entity_to_chunks_.clear();
    entity_cooccurrence_.clear();

    uint32_t crc_state = CRC32_INIT;
    auto fail = [this](const char* what) {
        EVDB_LOG_ERROR("KnowledgeGraph: Corrupt file (%s) — refusing to load", what);
        entity_to_chunks_.clear();
        entity_cooccurrence_.clear();
        return false;
    };

    // Read entity_to_chunks
    uint32_t entity_count;
    file.read(reinterpret_cast<char*>(&entity_count), 4);
    if (!file.good()) return fail("entity count");
    crc_state = crc32Update(crc_state, &entity_count, 4);

    for (uint32_t i = 0; i < entity_count; i++) {
        std::string name;
        if (!crcReadString(file, crc_state, total_size, name)) return fail("entity name");

        uint32_t chunk_count;
        file.read(reinterpret_cast<char*>(&chunk_count), 4);
        if (!file.good()) return fail("chunk count");
        crc_state = crc32Update(crc_state, &chunk_count, 4);
        if (!persist::boundedCount(chunk_count, 8, total_size)) return fail("chunk count bound");

        std::vector<uint64_t> chunks(chunk_count);
        for (uint32_t j = 0; j < chunk_count; j++) {
            file.read(reinterpret_cast<char*>(&chunks[j]), 8);
            if (!file.good()) return fail("chunk id");
            crc_state = crc32Update(crc_state, &chunks[j], 8);
        }
        entity_to_chunks_[name] = chunks;
    }

    // Read co-occurrence
    uint32_t cooc_count;
    file.read(reinterpret_cast<char*>(&cooc_count), 4);
    if (!file.good()) return fail("cooc count");
    crc_state = crc32Update(crc_state, &cooc_count, 4);

    for (uint32_t i = 0; i < cooc_count; i++) {
        std::string name;
        if (!crcReadString(file, crc_state, total_size, name)) return fail("cooc name");

        uint32_t nb_count;
        file.read(reinterpret_cast<char*>(&nb_count), 4);
        if (!file.good()) return fail("neighbor count");
        crc_state = crc32Update(crc_state, &nb_count, 4);
        if (!persist::boundedCount(nb_count, 4, total_size)) return fail("neighbor count bound");

        for (uint32_t j = 0; j < nb_count; j++) {
            std::string nb;
            if (!crcReadString(file, crc_state, total_size, nb)) return fail("neighbor name");
            entity_cooccurrence_[name].insert(nb);
        }
    }

    uint32_t stored_crc;
    file.read(reinterpret_cast<char*>(&stored_crc), 4);
    if (!file.good() || crc32Finalize(crc_state) != stored_crc) return fail("CRC mismatch");

    EVDB_LOG_INFO("KnowledgeGraph: Loaded %u entities", entity_count);
    return true;
}

} // namespace edgevdb
