#include "page_index.hpp"
#include <fstream>
#include <cstring>

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

    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), 8);

    chunk_to_page_.clear();
    doc_to_chunks_.clear();

    for (uint64_t i = 0; i < count; i++) {
        uint64_t chunk_id;
        uint32_t doc_id, page_number;
        file.read(reinterpret_cast<char*>(&chunk_id), 8);
        file.read(reinterpret_cast<char*>(&doc_id), 4);
        file.read(reinterpret_cast<char*>(&page_number), 4);

        if (!file.good()) {
            EVDB_LOG_ERROR("PageIndex: Read error at entry %llu", (unsigned long long)i);
            return false;
        }

        chunk_to_page_[chunk_id] = {doc_id, page_number};
        doc_to_chunks_[doc_id].push_back(chunk_id);
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

    std::ofstream file(file_path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        EVDB_LOG_ERROR("PageIndex: Cannot open %s for writing", file_path_.c_str());
        return false;
    }

    file.write(MAGIC, 8);
    file.write(reinterpret_cast<const char*>(&VERSION), 4);

    uint64_t count = chunk_to_page_.size();
    file.write(reinterpret_cast<const char*>(&count), 8);

    std::vector<uint8_t> crc_data;
    crc_data.reserve(count * 16);

    for (const auto& [chunk_id, page_info] : chunk_to_page_) {
        file.write(reinterpret_cast<const char*>(&chunk_id), 8);
        file.write(reinterpret_cast<const char*>(&page_info.first), 4);
        file.write(reinterpret_cast<const char*>(&page_info.second), 4);

        const uint8_t* p;
        p = reinterpret_cast<const uint8_t*>(&chunk_id);
        crc_data.insert(crc_data.end(), p, p + 8);
        p = reinterpret_cast<const uint8_t*>(&page_info.first);
        crc_data.insert(crc_data.end(), p, p + 4);
        p = reinterpret_cast<const uint8_t*>(&page_info.second);
        crc_data.insert(crc_data.end(), p, p + 4);
    }

    uint32_t crc = computeCRC32(crc_data.data(), crc_data.size());
    file.write(reinterpret_cast<const char*>(&crc), 4);

    EVDB_LOG_INFO("PageIndex: Saved %zu entries", chunk_to_page_.size());
    return file.good();
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

uint32_t PageIndex::computeCRC32(const uint8_t* data, size_t len) const {
    static const uint32_t crc_table[256] = {
        0x00000000,0x77073096,0xEE0E612C,0x990951BA,0x076DC419,0x706AF48F,0xE963A535,0x9E6495A3,
        0x0EDB8832,0x79DCB8A4,0xE0D5E91B,0x97D2D988,0x09B64C2B,0x7EB17CBE,0xE7B82D09,0x90BF1D9F,
        0x1DB71064,0x6AB020F2,0xF3B97148,0x84BE41DE,0x1ADAD47D,0x6DDDE4EB,0xF4D4B551,0x83D385C7,
        0x136C9856,0x646BA8C0,0xFD62F97A,0x8A65C9EC,0x14015C4F,0x63066CD9,0xFA0F3D63,0x8D080DF5,
        0x3B6E20C8,0x4C69105E,0xD56041E4,0xA2677172,0x3C03E4D1,0x4B04D447,0xD20D85FD,0xA50AB56B,
        0x35B5A8FA,0x42B2986C,0xDBBBC9D6,0xACBCF940,0x32D86CE3,0x45DF5C75,0xDCD60DCF,0xABD13D59,
        0x26D930AC,0x51DE003A,0xC8D75180,0xBFD06116,0x21B4F0B5,0x56B3C423,0xCFBA9599,0xB8BDA50F,
        0x2802B89E,0x5F058808,0xC60CD9B2,0xB10BE924,0x2F6F7C87,0x58684C11,0xC1611DAB,0xB6662D3D,
        0x76DC4190,0x01DB7106,0x98D220BC,0xEFD5102A,0x71B18589,0x06B6B51F,0x9FBFE4A5,0xE8B8D433,
        0x7807C9A2,0x0F00F934,0x9609A88E,0xE10E9818,0x7F6A0DBB,0x086D3D2D,0x91646C97,0xE6635C01,
        0x6B6B51F4,0x1C6C6162,0x856530D8,0xF262004E,0x6C0695ED,0x1B01A57B,0x8208F4C1,0xF50FC457,
        0x65B0D9C6,0x12B7E950,0x8BBEB8EA,0xFCB9887C,0x62DD1DDF,0x15DA2D49,0x8CD37CF3,0xFBD44C65,
        0x4DB26158,0x3AB551CE,0xA3BC0074,0xD4BB30E2,0x4ADFA541,0x3DD895D7,0xA4D1C46D,0xD3D6F4FB,
        0x4369E96A,0x346ED9FC,0xAD678846,0xDA60B8D0,0x44042D73,0x33031DE5,0xAA0A4C5F,0xDD0D7822,
        0x9B64C2B0,0xEC63F226,0x756AA39C,0x026D930A,0x9C0906A9,0xEB0E363F,0x72076785,0x05005713,
        0x95BF4A82,0xE2B87A14,0x7BB12BAE,0x0CB61B38,0x92D28E9B,0xE5D5BE0D,0x7CDCEFB7,0x0BDBDF21,
        0x86D3D2D4,0xF1D4E242,0x68DDB3F6,0x1FDA836E,0x81BE16CD,0xF6B9265B,0x6FB077E1,0x18B74777,
        0x88085AE6,0xFF0F6B70,0x66063BCA,0x11010B5C,0x8F659EFF,0xF862AE69,0x616BFFD3,0x166CCF45,
        0xA00AE278,0xD70DD2EE,0x4E048354,0x3903B3C2,0xA7672661,0xD06016F7,0x4969474D,0x3E6E77DB,
        0xAED16A4A,0xD9D65ADC,0x40DF0B66,0x37D83BF0,0xA9BCAE53,0xDEBB9EC5,0x47B2CF7F,0x30B5FFE9,
        0xBDBDF21C,0xCABAC28A,0x53B39330,0x24B4A3A6,0xBAD03605,0xCDD706FF,0x54DE5729,0x23D967BF,
        0xB3667A2E,0xC4614AB8,0x5D681B02,0x2A6F2B94,0xB40BBE37,0xC30C8EA1,0x5A05DF1B,0x2D02EF8D
    };
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc = crc_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

} // namespace edgevdb
