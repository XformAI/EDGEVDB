#include "object_store.hpp"
#include <fstream>
#include <algorithm>

namespace edgevdb {

ObjectStore::ObjectStore() : next_id_(1) {}

ObjectStore::ObjectStore(const std::string& file_path) : file_path_(file_path), next_id_(1) {}

bool ObjectStore::open() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return true;

    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
        EVDB_LOG_INFO("ObjectStore: No existing file, starting empty");
        return true;
    }

    char magic[8];
    file.read(magic, 8);
    const char expected[8] = {'E','V','D','B','O','B','J','\0'};
    if (std::memcmp(magic, expected, 8) != 0) {
        EVDB_LOG_ERROR("ObjectStore: Invalid magic");
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);

    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), 8);

    records_.clear();
    uint64_t max_id = 0;
    for (uint64_t i = 0; i < count; i++) {
        ObjectRecord record;
        file.read(reinterpret_cast<char*>(&record), sizeof(ObjectRecord));
        if (!file.good()) {
            EVDB_LOG_ERROR("ObjectStore: Read error at record %llu", (unsigned long long)i);
            return false;
        }
        records_[record.id] = record;
        if (record.id >= max_id) max_id = record.id;
    }

    next_id_ = max_id + 1;
    rebuildIndices();

    EVDB_LOG_INFO("ObjectStore: Loaded %zu records", records_.size());
    return true;
}

bool ObjectStore::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return false;

    std::ofstream file(file_path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return false;

    const char magic[8] = {'E','V','D','B','O','B','J','\0'};
    file.write(magic, 8);

    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), 4);

    uint64_t count = records_.size();
    file.write(reinterpret_cast<const char*>(&count), 8);

    std::vector<uint8_t> crc_data;
    for (const auto& [id, record] : records_) {
        file.write(reinterpret_cast<const char*>(&record), sizeof(ObjectRecord));
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&record);
        crc_data.insert(crc_data.end(), p, p + sizeof(ObjectRecord));
    }

    uint32_t crc = computeCRC32(crc_data.data(), crc_data.size());
    file.write(reinterpret_cast<const char*>(&crc), 4);

    EVDB_LOG_INFO("ObjectStore: Saved %zu records", records_.size());
    return file.good();
}

uint64_t ObjectStore::put(const std::string& type_name, const nlohmann::json& properties) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    ObjectRecord record;
    uint64_t id;

    // Check if update (properties has "id")
    if (properties.contains("id") && properties["id"].is_number()) {
        id = properties["id"].get<uint64_t>();
        auto it = records_.find(id);
        if (it != records_.end()) {
            record = it->second;
            removeFromIndices(id, std::string(record.type_name));
        } else {
            record.id = id;
            if (id >= next_id_) next_id_ = id + 1;
        }
    } else {
        id = next_id_++;
        record.id = id;
    }

    std::strncpy(record.type_name, type_name.c_str(), 63);
    record.type_name[63] = '\0';

    // Serialize properties to JSON string in data field
    std::string json_str = properties.dump();
    if (json_str.size() >= MAX_OBJECT_DATA - 1) {
        json_str = json_str.substr(0, MAX_OBJECT_DATA - 2);
    }
    std::memset(record.data, 0, MAX_OBJECT_DATA);
    std::memcpy(record.data, json_str.c_str(), json_str.size());

    auto now = std::chrono::system_clock::now();
    uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    if (record.created_at == 0) record.created_at = now_ms;
    record.updated_at = now_ms;
    record.is_deleted = false;

    records_[id] = record;
    indexRecord(id, type_name, properties);

    return id;
}

bool ObjectStore::get(uint64_t id, nlohmann::json& out) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = records_.find(id);
    if (it == records_.end() || it->second.is_deleted) return false;

    std::string json_str(reinterpret_cast<const char*>(it->second.data));
    try {
        out = nlohmann::json::parse(json_str);
        out["id"] = id;
        out["_type"] = std::string(it->second.type_name);
    } catch (...) {
        return false;
    }
    return true;
}

bool ObjectStore::remove(uint64_t id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = records_.find(id);
    if (it == records_.end()) return false;

    it->second.is_deleted = true;
    it->second.sync_vector_clock++;
    auto now = std::chrono::system_clock::now();
    it->second.updated_at = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    removeFromIndices(id, std::string(it->second.type_name));
    return true;
}

std::vector<nlohmann::json> ObjectStore::query(const std::string& type_name,
                                                 const std::string& property,
                                                 const std::string& value,
                                                 int limit) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    std::vector<nlohmann::json> results;

    auto type_it = property_index_.find(type_name);
    if (type_it == property_index_.end()) return results;

    auto prop_it = type_it->second.find(property);
    if (prop_it == type_it->second.end()) return results;

    auto val_it = prop_it->second.find(value);
    if (val_it == prop_it->second.end()) return results;

    for (uint64_t id : val_it->second) {
        if (static_cast<int>(results.size()) >= limit) break;
        auto rec_it = records_.find(id);
        if (rec_it != records_.end() && !rec_it->second.is_deleted) {
            nlohmann::json obj;
            std::string json_str(reinterpret_cast<const char*>(rec_it->second.data));
            try {
                obj = nlohmann::json::parse(json_str);
                obj["id"] = id;
                results.push_back(obj);
            } catch (...) {}
        }
    }
    return results;
}

std::vector<nlohmann::json> ObjectStore::getAll(const std::string& type_name,
                                                  int offset, int limit) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    std::vector<nlohmann::json> results;

    auto type_it = type_index_.find(type_name);
    if (type_it == type_index_.end()) return results;

    int skip = 0;
    for (uint64_t id : type_it->second) {
        auto rec_it = records_.find(id);
        if (rec_it == records_.end() || rec_it->second.is_deleted) continue;

        if (skip < offset) { skip++; continue; }
        if (static_cast<int>(results.size()) >= limit) break;

        nlohmann::json obj;
        std::string json_str(reinterpret_cast<const char*>(rec_it->second.data));
        try {
            obj = nlohmann::json::parse(json_str);
            obj["id"] = id;
            results.push_back(obj);
        } catch (...) {}
    }
    return results;
}

std::vector<uint64_t> ObjectStore::getIdsForType(const std::string& type_name) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = type_index_.find(type_name);
    if (it == type_index_.end()) return {};
    return it->second;
}

size_t ObjectStore::count(const std::string& type_name) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = type_index_.find(type_name);
    if (it == type_index_.end()) return 0;
    size_t c = 0;
    for (uint64_t id : it->second) {
        auto rec_it = records_.find(id);
        if (rec_it != records_.end() && !rec_it->second.is_deleted) c++;
    }
    return c;
}

size_t ObjectStore::totalCount() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    size_t c = 0;
    for (const auto& [id, rec] : records_) {
        if (!rec.is_deleted) c++;
    }
    return c;
}

bool ObjectStore::getRecord(uint64_t id, ObjectRecord& out) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = records_.find(id);
    if (it == records_.end()) return false;
    out = it->second;
    return true;
}

void ObjectStore::putRecord(const ObjectRecord& record) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    records_[record.id] = record;
    if (record.id >= next_id_) next_id_ = record.id + 1;
    // Rebuild indices for this record
    if (!record.is_deleted) {
        std::string json_str(reinterpret_cast<const char*>(record.data));
        try {
            auto props = nlohmann::json::parse(json_str);
            indexRecord(record.id, std::string(record.type_name), props);
        } catch (...) {}
    }
}

void ObjectStore::forEach(std::function<void(const ObjectRecord&)> fn) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    for (const auto& [id, rec] : records_) {
        fn(rec);
    }
}

void ObjectStore::rebuildIndices() {
    type_index_.clear();
    property_index_.clear();
    for (const auto& [id, record] : records_) {
        if (record.is_deleted) continue;
        std::string type(record.type_name);
        type_index_[type].push_back(id);

        std::string json_str(reinterpret_cast<const char*>(record.data));
        try {
            auto props = nlohmann::json::parse(json_str);
            indexRecord(id, type, props);
        } catch (...) {}
    }
}

void ObjectStore::indexRecord(uint64_t id, const std::string& type_name,
                               const nlohmann::json& props) {
    // Add to type index if not already there
    auto& type_ids = type_index_[type_name];
    if (std::find(type_ids.begin(), type_ids.end(), id) == type_ids.end()) {
        type_ids.push_back(id);
    }

    // Index all string and number properties
    if (props.is_object()) {
        for (const auto& [key, value] : props.items()) {
            std::string val_str;
            if (value.is_string()) {
                val_str = value.get<std::string>();
            } else if (value.is_number()) {
                val_str = value.dump();
            } else {
                continue;
            }
            property_index_[type_name][key][val_str].push_back(id);
        }
    }
}

void ObjectStore::removeFromIndices(uint64_t id, const std::string& type_name) {
    auto type_it = type_index_.find(type_name);
    if (type_it != type_index_.end()) {
        auto& ids = type_it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
    }

    auto pi = property_index_.find(type_name);
    if (pi != property_index_.end()) {
        for (auto& [prop, val_map] : pi->second) {
            for (auto& [val, ids] : val_map) {
                ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
            }
        }
    }
}

uint32_t ObjectStore::computeCRC32(const uint8_t* data, size_t len) const {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return crc ^ 0xFFFFFFFF;
}

} // namespace edgevdb
