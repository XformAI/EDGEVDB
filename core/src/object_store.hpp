#pragma once

#include "schema.hpp"
#include "log.hpp"
#include <nlohmann/json.hpp>

#include <unordered_map>
#include <shared_mutex>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <functional>

namespace edgevdb {

class ObjectStore {
public:
    ObjectStore();
    explicit ObjectStore(const std::string& file_path);

    bool open();
    bool save();

    uint64_t put(const std::string& type_name, const nlohmann::json& properties);
    bool get(uint64_t id, nlohmann::json& out) const;
    bool remove(uint64_t id);

    std::vector<nlohmann::json> query(const std::string& type_name,
                                       const std::string& property,
                                       const std::string& value,
                                       int limit = 100) const;

    std::vector<nlohmann::json> getAll(const std::string& type_name,
                                        int offset = 0, int limit = 100) const;

    std::vector<uint64_t> getIdsForType(const std::string& type_name) const;

    size_t count(const std::string& type_name) const;
    size_t totalCount() const;

    // Access for sync
    bool getRecord(uint64_t id, ObjectRecord& out) const;
    void putRecord(const ObjectRecord& record);
    void forEach(std::function<void(const ObjectRecord&)> fn) const;

private:
    std::unordered_map<uint64_t, ObjectRecord> records_;
    std::unordered_map<std::string, std::vector<uint64_t>> type_index_;
    std::unordered_map<std::string,
        std::unordered_map<std::string,
            std::unordered_map<std::string, std::vector<uint64_t>>>> property_index_;
    std::string file_path_;
    mutable std::shared_mutex rw_mutex_;
    uint64_t next_id_;

    void rebuildIndices();
    void indexRecord(uint64_t id, const std::string& type_name, const nlohmann::json& props);
    void removeFromIndices(uint64_t id, const std::string& type_name);
};

} // namespace edgevdb
