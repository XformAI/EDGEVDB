#include "chunk_store.hpp"
#include "crc32.hpp"
#include "persist.hpp"

#include <fstream>
#include <chrono>
#include <cstring>
#include <mutex>

namespace edgevdb {

ChunkStore::ChunkStore() : next_id_(1) {}

ChunkStore::ChunkStore(const std::string& file_path)
    : file_path_(file_path), next_id_(1) {}

bool ChunkStore::open() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return true; // in-memory only

    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
        EVDB_LOG_INFO("ChunkStore: No existing file at %s, starting empty", file_path_.c_str());
        return true; // file doesn't exist yet, empty store is valid
    }

    const uint64_t total_size = persist::fileSize(file);
    constexpr uint64_t HEADER_SIZE = 8 + 4 + 8;
    if (total_size < HEADER_SIZE + 4) {
        EVDB_LOG_ERROR("ChunkStore: File %s too small (%llu bytes)",
                       file_path_.c_str(), (unsigned long long)total_size);
        return false;
    }

    char magic[8];
    file.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0) {
        EVDB_LOG_ERROR("ChunkStore: Invalid magic in %s", file_path_.c_str());
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != VERSION) {
        EVDB_LOG_ERROR("ChunkStore: Unsupported version %u (expected %u) — rebuild the store",
                       version, VERSION);
        return false;
    }

    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), 8);
    if (!persist::boundedCount(count, sizeof(ChunkNode), total_size - HEADER_SIZE - 4)) {
        EVDB_LOG_ERROR("ChunkStore: Corrupt count %llu in %s",
                       (unsigned long long)count, file_path_.c_str());
        return false;
    }

    chunks_.clear();
    uint64_t max_id = 0;
    uint32_t crc_state = CRC32_INIT;
    for (uint64_t i = 0; i < count; i++) {
        ChunkNode chunk;
        file.read(reinterpret_cast<char*>(&chunk), sizeof(ChunkNode));
        if (!file.good()) {
            EVDB_LOG_ERROR("ChunkStore: Failed reading chunk %llu", (unsigned long long)i);
            chunks_.clear();
            return false;
        }
        crc_state = crc32Update(crc_state, &chunk, sizeof(ChunkNode));
        chunks_[chunk.id] = chunk;
        if (chunk.id >= max_id) max_id = chunk.id;
    }

    uint32_t stored_crc;
    file.read(reinterpret_cast<char*>(&stored_crc), 4);
    if (!file.good() || crc32Finalize(crc_state) != stored_crc) {
        EVDB_LOG_ERROR("ChunkStore: CRC32 mismatch in %s — refusing corrupt data",
                       file_path_.c_str());
        chunks_.clear();
        return false;
    }

    if (max_id + 1 > next_id_) next_id_ = max_id + 1;
    EVDB_LOG_INFO("ChunkStore: Loaded %zu chunks from %s", chunks_.size(), file_path_.c_str());
    return true;
}

bool ChunkStore::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) {
        EVDB_LOG_ERROR("ChunkStore: No file path set");
        return false;
    }

    bool ok = persist::atomicSave(file_path_, [this](std::ofstream& file) {
        file.write(MAGIC, 8);
        file.write(reinterpret_cast<const char*>(&VERSION), 4);

        uint64_t count = chunks_.size();
        file.write(reinterpret_cast<const char*>(&count), 8);

        uint32_t crc_state = CRC32_INIT;
        for (const auto& [id, chunk] : chunks_) {
            file.write(reinterpret_cast<const char*>(&chunk), sizeof(ChunkNode));
            crc_state = crc32Update(crc_state, &chunk, sizeof(ChunkNode));
        }

        uint32_t crc = crc32Finalize(crc_state);
        file.write(reinterpret_cast<const char*>(&crc), 4);
        return file.good();
    });

    if (ok) {
        EVDB_LOG_INFO("ChunkStore: Saved %zu chunks to %s", chunks_.size(), file_path_.c_str());
    }
    return ok;
}

uint64_t ChunkStore::put(const ChunkNode& chunk) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    ChunkNode new_chunk = chunk;
    new_chunk.id = next_id_++;

    auto now = std::chrono::system_clock::now();
    new_chunk.insert_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    chunks_[new_chunk.id] = new_chunk;
    return new_chunk.id;
}

uint64_t ChunkStore::putWithId(uint64_t id, const ChunkNode& chunk) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    ChunkNode new_chunk = chunk;
    new_chunk.id = id;
    chunks_[id] = new_chunk;
    if (id >= next_id_) next_id_ = id + 1;
    return id;
}

void ChunkStore::setIdBase(uint64_t base) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (base + 1 > next_id_) next_id_ = base + 1;
}

bool ChunkStore::get(uint64_t id, ChunkNode& out) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = chunks_.find(id);
    if (it == chunks_.end()) return false;
    out = it->second;
    return true;
}

bool ChunkStore::remove(uint64_t id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    return chunks_.erase(id) > 0;
}

size_t ChunkStore::size() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    return chunks_.size();
}

void ChunkStore::forEach(std::function<void(const ChunkNode&)> fn) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    for (const auto& [id, chunk] : chunks_) {
        fn(chunk);
    }
}

std::vector<uint64_t> ChunkStore::getAllIds() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    std::vector<uint64_t> ids;
    ids.reserve(chunks_.size());
    for (const auto& [id, chunk] : chunks_) {
        ids.push_back(id);
    }
    return ids;
}

bool ChunkStore::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    chunks_.clear();
    next_id_ = 1;
    return true;
}

} // namespace edgevdb
