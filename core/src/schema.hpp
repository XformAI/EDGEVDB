#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace edgevdb {

// ── Constants ──────────────────────────────────────────────
constexpr size_t EMBEDDING_DIM = 384;
constexpr size_t MAX_TEXT_LEN = 512;
constexpr size_t MAX_OBJECT_DATA = 4096;
constexpr size_t DEFAULT_TOP_K = 5;
constexpr size_t HNSW_OVER_FETCH_FACTOR = 3;

// ── 1. ChunkNode ───────────────────────────────────────────
struct ChunkNode {
    uint64_t id;                        // unique chunk identifier
    char text[512];                     // null-terminated UTF-8 chunk text
    float embedding[384];               // L2-normalised embedding vector
    uint32_t doc_id;                    // source document identifier
    uint32_t page_number;               // page within document (0-indexed)
    uint64_t insert_timestamp;          // Unix millis at insert time
    uint8_t reserved[16];              // padding for future fields, zero-init

    ChunkNode() {
        std::memset(this, 0, sizeof(ChunkNode));
    }
};

// ── 2. ObjectRecord ────────────────────────────────────────
struct ObjectRecord {
    uint64_t id;                        // unique object identifier
    char type_name[64];                 // object type/class name
    uint8_t data[4096];                 // serialized property blob (JSON string)
    uint64_t created_at;               // Unix millis
    uint64_t updated_at;               // Unix millis
    uint64_t sync_vector_clock;        // for CRDT sync
    bool is_deleted;                   // soft delete tombstone

    ObjectRecord() {
        std::memset(this, 0, sizeof(ObjectRecord));
    }
};

// ── 3. RelationEdge ────────────────────────────────────────
struct RelationEdge {
    uint64_t from_id;
    uint64_t to_id;
    char relation_name[64];
    uint64_t created_at;

    RelationEdge() {
        std::memset(this, 0, sizeof(RelationEdge));
    }
};

// ── 4. SyncEntry ───────────────────────────────────────────
struct SyncEntry {
    uint64_t record_id;
    char record_type[16];             // "chunk" or "object"
    uint64_t vector_clock;
    uint8_t operation;                // 0=insert, 1=update, 2=delete
    uint8_t payload[8192];            // serialized record

    SyncEntry() {
        std::memset(this, 0, sizeof(SyncEntry));
    }
};

// ── 5. QueryResult ─────────────────────────────────────────
struct QueryResult {
    uint64_t chunk_id;
    float score;
    float cosine_score;
    float page_score;
    float keyword_score;
    uint32_t doc_id;
    uint32_t page_number;
    char text[512];

    QueryResult() {
        std::memset(this, 0, sizeof(QueryResult));
    }
};

// ── Static asserts ─────────────────────────────────────────
static_assert(std::is_trivially_copyable<ChunkNode>::value,
    "ChunkNode must be trivially copyable");
static_assert(std::is_trivially_copyable<ObjectRecord>::value,
    "ObjectRecord must be trivially copyable");
static_assert(std::is_trivially_copyable<RelationEdge>::value,
    "RelationEdge must be trivially copyable");
static_assert(std::is_trivially_copyable<SyncEntry>::value,
    "SyncEntry must be trivially copyable");
static_assert(std::is_trivially_copyable<QueryResult>::value,
    "QueryResult must be trivially copyable");

} // namespace edgevdb
