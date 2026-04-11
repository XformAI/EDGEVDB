#pragma once

#include "schema.hpp"
#include "log.hpp"

#include <unordered_map>
#include <shared_mutex>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <cmath>

namespace edgevdb {

class PageIndex {
public:
    PageIndex();
    explicit PageIndex(const std::string& file_path);

    bool open();
    bool save();

    void insert(uint64_t chunk_id, uint32_t doc_id, uint32_t page_number);
    bool getPage(uint64_t chunk_id, uint32_t& doc_id_out, uint32_t& page_out) const;
    std::vector<uint64_t> getChunksForDoc(uint32_t doc_id) const;
    float computePageProximity(uint64_t query_chunk_id, uint64_t neighbour_chunk_id) const;
    bool remove(uint64_t chunk_id);
    void clear();

private:
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> chunk_to_page_;
    std::unordered_map<uint32_t, std::vector<uint64_t>> doc_to_chunks_;
    std::string file_path_;
    mutable std::shared_mutex rw_mutex_;

    static constexpr char MAGIC[8] = {'E','V','D','B','P','A','G','\0'};
    static constexpr uint32_t VERSION = 1;

    uint32_t computeCRC32(const uint8_t* data, size_t len) const;
};

} // namespace edgevdb
