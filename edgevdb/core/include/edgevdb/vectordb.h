#ifndef VECTORDB_H
#define VECTORDB_H

/**
 * @file vectordb.h
 * @brief EdgeVDB SDK — Stable Public C API
 * @version 1.0.0
 *
 * This is the ONLY header external consumers need to include.
 * It provides a stable ABI using opaque handle types.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Handle types ──────────────────────────────────────── */

/** Main database handle */
typedef struct EvdbHandle_       EvdbHandle;

/** Query result handle */
typedef struct EvdbQueryHandle_  EvdbQueryHandle;

/** Embedder handle */
typedef struct EvdbEmbedder_     EvdbEmbedder;

/** Sync engine handle */
typedef struct EvdbSyncEngine_   EvdbSyncEngine;

/* ── Error codes ───────────────────────────────────────── */

/** Error codes returned by EdgeVDB functions */
typedef enum {
    EVDB_OK              = 0,  /**< Operation succeeded */
    EVDB_ERR_NULL_HANDLE = 1,  /**< Null handle passed */
    EVDB_ERR_IO          = 2,  /**< I/O or internal error */
    EVDB_ERR_OOM         = 3,  /**< Out of memory */
    EVDB_ERR_NOT_FOUND   = 4,  /**< Record not found */
    EVDB_ERR_INVALID_ARG = 5,  /**< Invalid argument */
    EVDB_ERR_ONNX        = 6,  /**< ONNX Runtime error */
    EVDB_ERR_SYNC        = 7,  /**< Sync error */
} EvdbError;

/* ── Config struct ─────────────────────────────────────── */

/** Configuration for opening an EdgeVDB instance */
typedef struct {
    const char* storage_dir;         /**< Directory for .bin files */
    int hnsw_M;                      /**< HNSW M parameter, default 16 */
    int hnsw_ef_construction;        /**< HNSW ef_construction, default 200 */
    int hnsw_ef_search;              /**< HNSW ef_search, default 64 */
    float ranker_alpha;              /**< Cosine weight, default 0.70 */
    float ranker_beta;               /**< Page proximity weight, default 0.20 */
    float ranker_gamma;              /**< Keyword weight, default 0.10 */
    int token_budget;                /**< Max tokens in context, default 3200 */
    int embedding_threads;           /**< ONNX thread count, default 2 */
    int enable_knowledge_graph;      /**< 0 or 1, default 1 */
    int enable_sync;                 /**< 0 or 1, default 0 */
    const char* device_id;           /**< For sync, NULL = auto-generate */
} EvdbConfig;

/* ── Lifecycle ─────────────────────────────────────────── */

/**
 * @brief Open an EdgeVDB instance.
 * @param config Configuration. Use evdb_default_config() to fill defaults.
 * @return Handle on success, NULL on failure.
 */
EvdbHandle* evdb_open(const EvdbConfig* config);

/** Close and free an EdgeVDB instance. Saves data before closing. */
void evdb_close(EvdbHandle* h);

/** Save all data to disk. */
EvdbError evdb_save(EvdbHandle* h);

/** Fill config with safe default values. */
void evdb_default_config(EvdbConfig* out);

/* ── Embedding pipeline ────────────────────────────────── */

/** Create an embedder from ONNX model and vocab files. */
EvdbEmbedder* evdb_embedder_create(const char* model_path, const char* vocab_path, int threads);

/** Destroy an embedder. */
void evdb_embedder_destroy(EvdbEmbedder* e);

/** Embed text into a 384-dim float vector. */
EvdbError evdb_embed_text(EvdbEmbedder* e, const char* text, float* out_384);

/* ── Vector store — insert ─────────────────────────────── */

/** Insert text (auto-embed) into the vector store. */
EvdbError evdb_insert_text(EvdbHandle* h, EvdbEmbedder* e,
                            const char* text, uint32_t doc_id, uint32_t page_number,
                            uint64_t* out_chunk_id);

/** Insert a pre-computed embedding. */
EvdbError evdb_insert_chunk(EvdbHandle* h,
                             const char* text, const float* embedding_384,
                             uint32_t doc_id, uint32_t page_number,
                             uint64_t* out_chunk_id);

/** Remove a chunk by ID. */
EvdbError evdb_remove_chunk(EvdbHandle* h, uint64_t chunk_id);

/* ── Vector store — query ──────────────────────────────── */

/** Query using text (auto-embed). Returns a query handle. */
EvdbQueryHandle* evdb_query_text(EvdbHandle* h, EvdbEmbedder* e,
                                  const char* query_text, int top_k,
                                  int use_kg_expansion);

/** Query using a pre-computed embedding. */
EvdbQueryHandle* evdb_query_vector(EvdbHandle* h,
                                    const float* query_embedding_384,
                                    const char* query_text_for_keyword,
                                    int top_k);

/** Get number of results. */
int evdb_result_count(EvdbQueryHandle* q);

/** Get result text at index. */
const char* evdb_result_text(EvdbQueryHandle* q, int index);

/** Get result score at index. */
float evdb_result_score(EvdbQueryHandle* q, int index);

/** Get result chunk ID at index. */
uint64_t evdb_result_chunk_id(EvdbQueryHandle* q, int index);

/** Get result page number at index. */
uint32_t evdb_result_page(EvdbQueryHandle* q, int index);

/** Get assembled RAG context string. */
const char* evdb_result_context_string(EvdbQueryHandle* q);

/** Free a query result handle. */
void evdb_query_free(EvdbQueryHandle* q);

/* ── Object store ──────────────────────────────────────── */

/** Put an object into the store. Returns object ID. */
EvdbError evdb_object_put(EvdbHandle* h,
                           const char* type_name,
                           const char* json_properties,
                           uint64_t* out_id);

/** Get an object by ID. Output is JSON string. */
EvdbError evdb_object_get(EvdbHandle* h, uint64_t id, char* out_json, int out_json_size);

/** Remove an object (soft delete). */
EvdbError evdb_object_remove(EvdbHandle* h, uint64_t id);

/** Query objects by type and property. */
EvdbError evdb_object_query(EvdbHandle* h,
                             const char* type_name,
                             const char* filter_property,
                             const char* filter_value,
                             char* out_json_array,
                             int out_size,
                             int limit);

/* ── Relations ─────────────────────────────────────────── */

/** Add a typed relation between two objects. */
EvdbError evdb_relation_add(EvdbHandle* h,
                             const char* relation_name,
                             uint64_t from_id, uint64_t to_id);

/** Get target IDs for a relation from a source. */
EvdbError evdb_relation_get_targets(EvdbHandle* h,
                                     const char* relation_name,
                                     uint64_t from_id,
                                     uint64_t* out_ids, int* inout_count);

/* ── Sync ──────────────────────────────────────────────── */

/** Create a sync engine. */
EvdbSyncEngine* evdb_sync_create(EvdbHandle* h, const char* device_id);

/** Destroy a sync engine. */
void evdb_sync_destroy(EvdbSyncEngine* s);

/** Export a sync delta since a given clock. */
EvdbError evdb_sync_export_delta(EvdbSyncEngine* s,
                                  uint64_t since_clock,
                                  char* out_json, int out_size);

/** Apply a received sync delta. */
EvdbError evdb_sync_apply_delta(EvdbSyncEngine* s,
                                 const char* delta_json,
                                 int* out_applied, int* out_skipped);

/** Export sync delta to a file. */
EvdbError evdb_sync_export_to_file(EvdbSyncEngine* s,
                                    const char* path, uint64_t since_clock);

/** Import sync delta from a file. */
EvdbError evdb_sync_import_from_file(EvdbSyncEngine* s, const char* path);

/** Get current sync clock value. */
uint64_t evdb_sync_current_clock(EvdbSyncEngine* s);

/* ── Utilities ─────────────────────────────────────────── */

/** Get version string. */
const char* evdb_version_string(void);

/** Get human-readable error string. */
const char* evdb_error_string(EvdbError err);

/** Set log verbosity (0=off, 1=error, 2=info, 3=debug). */
void evdb_set_log_level(int level);

#ifdef __cplusplus
}
#endif

#endif /* VECTORDB_H */
