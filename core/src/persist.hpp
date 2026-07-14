#pragma once

// Crash-safe persistence helpers shared by all stores.
//
// atomicSave() writes to "<path>.tmp" and replaces the target only after the
// full payload is on disk, so a crash mid-save never destroys the previous
// good file. Loaders use fileRemaining()/boundedCount() to validate on-disk
// counts before allocating, so a truncated or hostile file fails cleanly
// instead of causing huge allocations or partially-initialized state.

#include "log.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace edgevdb {
namespace persist {

// Create a directory (and parents) if it does not exist. Returns true when
// the directory exists afterwards.
inline bool ensureDir(const std::string& path) {
    if (path.empty()) return false;
    std::string partial;
    for (size_t i = 0; i <= path.size(); i++) {
        char c = (i < path.size()) ? path[i] : '/';
        if (c == '/' || c == '\\') {
            if (!partial.empty() && partial.back() != ':') {
#ifdef _WIN32
                CreateDirectoryA(partial.c_str(), nullptr);
#else
                mkdir(partial.c_str(), 0755);
#endif
            }
        }
        if (i < path.size()) partial += c;
    }
#ifdef _WIN32
    DWORD attrs = GetFileAttributesA(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

// Replace `target` with `tmp` atomically (as atomically as the OS allows).
inline bool atomicReplace(const std::string& tmp, const std::string& target) {
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), target.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        EVDB_LOG_ERROR("persist: MoveFileEx %s -> %s failed (err=%lu)",
                       tmp.c_str(), target.c_str(), GetLastError());
        std::remove(tmp.c_str());
        return false;
    }
    return true;
#else
    if (std::rename(tmp.c_str(), target.c_str()) != 0) {
        EVDB_LOG_ERROR("persist: rename %s -> %s failed", tmp.c_str(), target.c_str());
        std::remove(tmp.c_str());
        return false;
    }
    return true;
#endif
}

// Run `writer` against a temp file; on success, atomically replace `path`.
inline bool atomicSave(const std::string& path,
                       const std::function<bool(std::ofstream&)>& writer) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            EVDB_LOG_ERROR("persist: cannot open %s for writing", tmp.c_str());
            return false;
        }
        if (!writer(file)) {
            file.close();
            std::remove(tmp.c_str());
            return false;
        }
        file.flush();
        if (!file.good()) {
            file.close();
            std::remove(tmp.c_str());
            EVDB_LOG_ERROR("persist: write to %s failed", tmp.c_str());
            return false;
        }
    }
    return atomicReplace(tmp, path);
}

// Total size of an open input stream, preserving the current read position.
inline uint64_t fileSize(std::ifstream& file) {
    std::streampos cur = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos end = file.tellg();
    file.seekg(cur);
    return end < 0 ? 0 : static_cast<uint64_t>(end);
}

// True when `count` records of `record_size` bytes fit in `available` bytes.
inline bool boundedCount(uint64_t count, uint64_t record_size, uint64_t available) {
    if (record_size == 0) return true;
    return count <= available / record_size;
}

} // namespace persist
} // namespace edgevdb
