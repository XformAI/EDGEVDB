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
    // Insert preserving the caller-supplied id (used by sync to keep ids
    // identical across replicas). Overwrites any existing chunk with that id.
    uint64_t putWithId(uint64_t id, const ChunkNode& chunk);
    bool get(uint64_t id, ChunkNode& out) const;
    bool remove(uint64_t id);
    size_t size() const;
    void forEach(std::function<void(const ChunkNode&)> fn) const;
    std::vector<uint64_t> getAllIds() const;
    bool clear();

    // Base value for newly assigned ids; used to partition the id space
    // per device so replicas never collide (see SyncEngine).
    void setIdBase(uint64_t base);

private:
    std::unordered_map<uint64_t, ChunkNode> chunks_;
    std::string file_path_;
    mutable std::shared_mutex rw_mutex_;
    uint64_t next_id_;

    static constexpr char MAGIC[8] = {'E','V','D','B','C','H','K','\0'};
    // v2: CRC computed with the standard CRC-32 table and enforced on load.
    static constexpr uint32_t VERSION = 2;
};

} // namespace edgevdb
