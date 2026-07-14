#include "page_index.hpp"
#include "crc32.hpp"
#include "persist.hpp"
#include <fstream>
#include <cstring>
#include <mutex>

namespace edgevdb {

PageIndex::PageIndex() {}

PageIndex::PageIndex(const std::string& file_path) : file_path_(file_path) {}

bool PageIndex::open() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return true;

    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
        EVDB_LOG_INFO("PageIndex: No existing file, starting empty");
        return true;
    }

    char magic[8];
    file.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0) {
        EVDB_LOG_ERROR("PageIndex: Invalid magic");
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != VERSION) {
        EVDB_LOG_ERROR("PageIndex: Unsupported version %u", version);
        return false;
    }

    const uint64_t total_size = persist::fileSize(file);
    constexpr uint64_t HEADER_SIZE = 8 + 4 + 8;
    constexpr uint64_t ENTRY_SIZE = 8 + 4 + 4;

    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), 8);
    if (total_size < HEADER_SIZE + 4 ||
        !persist::boundedCount(count, ENTRY_SIZE, total_size - HEADER_SIZE - 4)) {
        EVDB_LOG_ERROR("PageIndex: Corrupt count %llu", (unsigned long long)count);
        return false;
    }

    chunk_to_page_.clear();
    doc_to_chunks_.clear();

    uint32_t crc_state = CRC32_INIT;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t chunk_id;
        uint32_t doc_id, page_number;
        file.read(reinterpret_cast<char*>(&chunk_id), 8);
        file.read(reinterpret_cast<char*>(&doc_id), 4);
        file.read(reinterpret_cast<char*>(&page_number), 4);

        if (!file.good()) {
            EVDB_LOG_ERROR("PageIndex: Read error at entry %llu", (unsigned long long)i);
            chunk_to_page_.clear();
            doc_to_chunks_.clear();
            return false;
        }

        crc_state = crc32Update(crc_state, &chunk_id, 8);
        crc_state = crc32Update(crc_state, &doc_id, 4);
        crc_state = crc32Update(crc_state, &page_number, 4);

        chunk_to_page_[chunk_id] = {doc_id, page_number};
        doc_to_chunks_[doc_id].push_back(chunk_id);
    }

    uint32_t stored_crc;
    file.read(reinterpret_cast<char*>(&stored_crc), 4);
    if (!file.good() || crc32Finalize(crc_state) != stored_crc) {
        EVDB_LOG_ERROR("PageIndex: CRC32 mismatch — refusing corrupt data");
        chunk_to_page_.clear();
        doc_to_chunks_.clear();
        return false;
    }

    // Sort doc_to_chunks_ by page number
    for (auto& [doc_id, chunks] : doc_to_chunks_) {
        std::sort(chunks.begin(), chunks.end(), [this](uint64_t a, uint64_t b) {
            return chunk_to_page_[a].second < chunk_to_page_[b].second;
        });
    }

    EVDB_LOG_INFO("PageIndex: Loaded %zu entries", chunk_to_page_.size());
    return true;
}

bool PageIndex::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return false;

    bool ok = persist::atomicSave(file_path_, [this](std::ofstream& file) {
        file.write(MAGIC, 8);
        file.write(reinterpret_cast<const char*>(&VERSION), 4);

        uint64_t count = chunk_to_page_.size();
        file.write(reinterpret_cast<const char*>(&count), 8);

        uint32_t crc_state = CRC32_INIT;
        for (const auto& [chunk_id, page_info] : chunk_to_page_) {
            file.write(reinterpret_cast<const char*>(&chunk_id), 8);
            file.write(reinterpret_cast<const char*>(&page_info.first), 4);
            file.write(reinterpret_cast<const char*>(&page_info.second), 4);
            crc_state = crc32Update(crc_state, &chunk_id, 8);
            crc_state = crc32Update(crc_state, &page_info.first, 4);
            crc_state = crc32Update(crc_state, &page_info.second, 4);
        }

        uint32_t crc = crc32Finalize(crc_state);
        file.write(reinterpret_cast<const char*>(&crc), 4);
        return file.good();
    });

    if (ok) EVDB_LOG_INFO("PageIndex: Saved %zu entries", chunk_to_page_.size());
    return ok;
}

void PageIndex::insert(uint64_t chunk_id, uint32_t doc_id, uint32_t page_number) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    chunk_to_page_[chunk_id] = {doc_id, page_number};
    doc_to_chunks_[doc_id].push_back(chunk_id);
}

bool PageIndex::getPage(uint64_t chunk_id, uint32_t& doc_id_out, uint32_t& page_out) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = chunk_to_page_.find(chunk_id);
    if (it == chunk_to_page_.end()) return false;
    doc_id_out = it->second.first;
    page_out = it->second.second;
    return true;
}

std::vector<uint64_t> PageIndex::getChunksForDoc(uint32_t doc_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = doc_to_chunks_.find(doc_id);
    if (it == doc_to_chunks_.end()) return {};
    return it->second;
}

float PageIndex::computePageProximity(uint64_t query_chunk_id, uint64_t neighbour_chunk_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    auto it_q = chunk_to_page_.find(query_chunk_id);
    auto it_n = chunk_to_page_.find(neighbour_chunk_id);
    if (it_q == chunk_to_page_.end() || it_n == chunk_to_page_.end()) return 0.0f;

    // Different documents → 0.0
    if (it_q->second.first != it_n->second.first) return 0.0f;

    // Same document: 1.0 / (1.0 + |page_a - page_b|)
    int page_diff = std::abs(static_cast<int>(it_q->second.second) -
                              static_cast<int>(it_n->second.second));
    return 1.0f / (1.0f + static_cast<float>(page_diff));
}

bool PageIndex::remove(uint64_t chunk_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = chunk_to_page_.find(chunk_id);
    if (it == chunk_to_page_.end()) return false;

    uint32_t doc_id = it->second.first;
    chunk_to_page_.erase(it);

    auto doc_it = doc_to_chunks_.find(doc_id);
    if (doc_it != doc_to_chunks_.end()) {
        auto& chunks = doc_it->second;
        chunks.erase(std::remove(chunks.begin(), chunks.end(), chunk_id), chunks.end());
        if (chunks.empty()) {
            doc_to_chunks_.erase(doc_it);
        }
    }
    return true;
}

void PageIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    chunk_to_page_.clear();
    doc_to_chunks_.clear();
}

} // namespace edgevdb
