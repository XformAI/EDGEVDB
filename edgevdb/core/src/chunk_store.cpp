#include "chunk_store.hpp"
#include <fstream>
#include <chrono>
#include <cstring>

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

    // Read magic
    char magic[8];
    file.read(magic, 8);
    if (std::memcmp(magic, MAGIC, 8) != 0) {
        EVDB_LOG_ERROR("ChunkStore: Invalid magic in %s", file_path_.c_str());
        return false;
    }

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != VERSION) {
        EVDB_LOG_ERROR("ChunkStore: Unsupported version %u", version);
        return false;
    }

    // Read count
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), 8);

    // Read chunks
    chunks_.clear();
    uint64_t max_id = 0;
    for (uint64_t i = 0; i < count; i++) {
        ChunkNode chunk;
        file.read(reinterpret_cast<char*>(&chunk), sizeof(ChunkNode));
        if (!file.good()) {
            EVDB_LOG_ERROR("ChunkStore: Failed reading chunk %llu", (unsigned long long)i);
            return false;
        }
        chunks_[chunk.id] = chunk;
        if (chunk.id >= max_id) max_id = chunk.id;
    }

    // Read and verify CRC32
    uint32_t stored_crc;
    file.read(reinterpret_cast<char*>(&stored_crc), 4);

    // Compute CRC32 over all chunk data
    std::vector<uint8_t> all_data;
    all_data.resize(count * sizeof(ChunkNode));
    size_t offset = 0;
    // Re-read chunk data for CRC verification
    file.seekg(8 + 4 + 8); // skip header
    file.read(reinterpret_cast<char*>(all_data.data()), count * sizeof(ChunkNode));
    uint32_t computed_crc = computeCRC32(all_data.data(), all_data.size());
    if (computed_crc != stored_crc) {
        EVDB_LOG_ERROR("ChunkStore: CRC32 mismatch (stored=%u, computed=%u)", stored_crc, computed_crc);
        // Continue anyway but log warning
    }

    next_id_ = max_id + 1;
    EVDB_LOG_INFO("ChunkStore: Loaded %zu chunks from %s", chunks_.size(), file_path_.c_str());
    return true;
}

bool ChunkStore::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) {
        EVDB_LOG_ERROR("ChunkStore: No file path set");
        return false;
    }

    std::ofstream file(file_path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        EVDB_LOG_ERROR("ChunkStore: Cannot open %s for writing", file_path_.c_str());
        return false;
    }

    // Write magic
    file.write(MAGIC, 8);

    // Write version
    file.write(reinterpret_cast<const char*>(&VERSION), 4);

    // Write count
    uint64_t count = chunks_.size();
    file.write(reinterpret_cast<const char*>(&count), 8);

    // Write chunks and accumulate CRC data
    std::vector<uint8_t> all_data;
    all_data.reserve(count * sizeof(ChunkNode));
    for (const auto& [id, chunk] : chunks_) {
        file.write(reinterpret_cast<const char*>(&chunk), sizeof(ChunkNode));
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&chunk);
        all_data.insert(all_data.end(), ptr, ptr + sizeof(ChunkNode));
    }

    // Write CRC32
    uint32_t crc = computeCRC32(all_data.data(), all_data.size());
    file.write(reinterpret_cast<const char*>(&crc), 4);

    file.flush();
    EVDB_LOG_INFO("ChunkStore: Saved %zu chunks to %s", chunks_.size(), file_path_.c_str());
    return file.good();
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

uint32_t ChunkStore::computeCRC32(const uint8_t* data, size_t len) const {
    // Standard CRC32 implementation (ISO 3309 / ITU-T V.42)
    static const uint32_t crc_table[256] = {
        0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
        0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91B, 0x97D2D988,
        0x09B64C2B, 0x7EB17CBE, 0xE7B82D09, 0x90BF1D9F, 0x1DB71064, 0x6AB020F2,
        0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
        0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
        0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
        0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
        0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
        0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F0B5, 0x56B3C423,
        0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
        0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
        0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
        0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D,
        0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
        0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
        0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
        0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
        0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
        0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7822, 0x3B6E20C8, 0x4C69105E,
        0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
        0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75,
        0xDCD60DCF, 0xABD13D59, 0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116,
        0x21B4F0B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808,
        0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
        0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F,
        0x9FBFE4A5, 0xE8B8D433, 0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818,
        0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162,
        0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
        0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49,
        0x8CD37CF3, 0xFBD44C65, 0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2,
        0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC,
        0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7822,
        0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
        0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
        0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
        0x68DDB3F6, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
        0x88085AE6, 0xFF0F6B70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69,
        0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
        0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
        0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
        0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD706FF,
        0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
        0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
    };

    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        uint8_t byte = data[i];
        crc = crc_table[(crc ^ byte) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

} // namespace edgevdb
