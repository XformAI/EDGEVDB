#pragma once

#include "schema.hpp"
#include "log.hpp"

#include <unordered_map>
#include <shared_mutex>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace edgevdb {

class ChunkStore {
public:
    ChunkStore();
    explicit ChunkStore(const std::string& file_path);

    bool open();
    bool save();

    uint64_t put(const ChunkNode& chunk);
    bool get(uint64_t id, ChunkNode& out) const;
    bool remove(uint64_t id);
    size_t size() const;
    void forEach(std::function<void(const ChunkNode&)> fn) const;
    std::vector<uint64_t> getAllIds() const;
    bool clear();

private:
    std::unordered_map<uint64_t, ChunkNode> chunks_;
    std::string file_path_;
    mutable std::shared_mutex rw_mutex_;
    uint64_t next_id_;

    static constexpr char MAGIC[8] = {'E','V','D','B','C','H','K','\0'};
    static constexpr uint32_t VERSION = 1;

    uint32_t computeCRC32(const uint8_t* data, size_t len) const;
};

} // namespace edgevdb
