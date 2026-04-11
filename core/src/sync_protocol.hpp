#pragma once

#include "schema.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace edgevdb {

struct SyncConfig {
    bool enabled = false;
    std::string device_id;
    std::string sync_endpoint;
    int sync_interval_seconds = 60;
    bool sync_chunks = true;
    bool sync_objects = true;
};

} // namespace edgevdb
