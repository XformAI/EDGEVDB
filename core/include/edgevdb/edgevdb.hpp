#pragma once

/**
 * @file edgevdb.hpp
 * @brief EdgeVDB C++ RAII wrapper (optional convenience header)
 */

#include "vectordb.h"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace edgevdb {

class EdgeVDBError : public std::runtime_error {
public:
    EvdbError code;
    EdgeVDBError(EvdbError c, const std::string& msg)
        : std::runtime_error(msg), code(c) {}
};

inline void checkError(EvdbError err) {
    if (err != EVDB_OK) {
        throw EdgeVDBError(err, evdb_error_string(err));
    }
}

class QueryResultsRAII {
    EvdbQueryHandle* handle_;
public:
    explicit QueryResultsRAII(EvdbQueryHandle* h) : handle_(h) {}
    ~QueryResultsRAII() { if (handle_) evdb_query_free(handle_); }
    QueryResultsRAII(const QueryResultsRAII&) = delete;
    QueryResultsRAII& operator=(const QueryResultsRAII&) = delete;

    int count() const { return evdb_result_count(handle_); }
    std::string text(int i) const { return evdb_result_text(handle_, i); }
    float score(int i) const { return evdb_result_score(handle_, i); }
    uint64_t chunkId(int i) const { return evdb_result_chunk_id(handle_, i); }
    uint32_t page(int i) const { return evdb_result_page(handle_, i); }
    std::string contextString() const { return evdb_result_context_string(handle_); }
    EvdbQueryHandle* get() const { return handle_; }
};

class EmbedderRAII {
    EvdbEmbedder* handle_;
public:
    EmbedderRAII(const std::string& model_path, const std::string& vocab_path, int threads = 2)
        : handle_(evdb_embedder_create(model_path.c_str(), vocab_path.c_str(), threads)) {}
    ~EmbedderRAII() { if (handle_) evdb_embedder_destroy(handle_); }
    EmbedderRAII(const EmbedderRAII&) = delete;
    EmbedderRAII& operator=(const EmbedderRAII&) = delete;

    EvdbEmbedder* get() const { return handle_; }
    bool valid() const { return handle_ != nullptr; }
};

class DatabaseRAII {
    EvdbHandle* handle_;
public:
    explicit DatabaseRAII(const EvdbConfig& config)
        : handle_(evdb_open(&config)) {}
    ~DatabaseRAII() { if (handle_) evdb_close(handle_); }
    DatabaseRAII(const DatabaseRAII&) = delete;
    DatabaseRAII& operator=(const DatabaseRAII&) = delete;

    EvdbHandle* get() const { return handle_; }
    bool valid() const { return handle_ != nullptr; }
    void save() { checkError(evdb_save(handle_)); }
};

} // namespace edgevdb
