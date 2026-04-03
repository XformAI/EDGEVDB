# EdgeVDB C API Reference

> Include: `#include "edgevdb/vectordb.h"`

## Lifecycle

### `void evdb_default_config(EvdbConfig* out)`
Fill a config struct with default values.

### `EvdbHandle* evdb_open(const EvdbConfig* config)`
Open or create a database. Returns `NULL` on failure.

### `void evdb_close(EvdbHandle* h)`
Close and free a database handle. **Call `evdb_save()` first** to ensure all data is flushed to disk. Do not rely on `evdb_close` to persist unsaved changes.

### `EvdbError evdb_save(EvdbHandle* h)`
Flush all data to disk.

## Embedding (Optional)

> The embedder is optional. Use `insert_chunk()` and `query_vector()` with pre-computed embeddings to avoid any ONNX dependency.

### `EvdbEmbedder* evdb_embedder_create(const char* model_path, const char* vocab_path, int threads)`
Create an embedder from ONNX model and vocab files. Returns `NULL` on failure.

### `void evdb_embedder_destroy(EvdbEmbedder* e)`
Free an embedder.

### `EvdbError evdb_embed_text(EvdbEmbedder* e, const char* text, float* out_384)`
Produce a 384-dim L2-normalized embedding.

## Vector Store — Insert

### `EvdbError evdb_insert_text(h, embedder, text, doc_id, page, &chunk_id)`
Auto-embed text and insert. Requires an embedder.

### `EvdbError evdb_insert_chunk(h, text, embedding_384, doc_id, page, &chunk_id)`
Insert with a pre-computed 384-dim embedding. **No embedder needed.**

### `EvdbError evdb_remove_chunk(h, chunk_id)`
Remove a chunk by ID.

## Vector Store — Query

### `EvdbQueryHandle* evdb_query_text(h, embedder, query, top_k, use_kg)`
Query with auto-embedding. Requires an embedder.

### `EvdbQueryHandle* evdb_query_vector(h, embedding_384, query_text_for_keyword, top_k)`
Query with a pre-computed embedding. **No embedder needed.**

### Result Accessors

| Function | Returns | Description |
|----------|---------|-------------|
| `evdb_result_count(q)` | `int` | Number of results |
| `evdb_result_text(q, i)` | `const char*` | Text at index |
| `evdb_result_score(q, i)` | `float` | Score at index |
| `evdb_result_chunk_id(q, i)` | `uint64_t` | Chunk ID at index |
| `evdb_result_page(q, i)` | `uint32_t` | Page number at index |
| `evdb_result_context_string(q)` | `const char*` | Assembled RAG context |

### `void evdb_query_free(q)`
Free query results. **Always call this after processing results.**

## Object Store

### `EvdbError evdb_object_put(h, type_name, json_properties, &id)`
Insert or update an object. Returns object ID.

### `EvdbError evdb_object_get(h, id, out_json, size)`
Get object as JSON string.

### `EvdbError evdb_object_remove(h, id)`
Soft-delete an object.

### `EvdbError evdb_object_query(h, type, prop, value, out, size, limit)`
Query objects by type and property filter.

## Relations

### `EvdbError evdb_relation_add(h, name, from_id, to_id)`
Add a typed directional relation.

### `EvdbError evdb_relation_get_targets(h, name, from_id, out_ids, &count)`
Get target IDs for a relation from a source. `count` is an **in-out** parameter: caller sets `*count` to the buffer capacity of `out_ids`; on return, `*count` is set to the actual number of targets written.

## Sync

### `EvdbSyncEngine* evdb_sync_create(h, device_id)`
Create a sync engine.

### `void evdb_sync_destroy(EvdbSyncEngine* s)`
Destroy a sync engine.

### `EvdbError evdb_sync_export_delta(s, since, out, size)`
Export changes since a clock value as JSON.

### `EvdbError evdb_sync_apply_delta(s, json, &applied, &skipped)`
Apply a received delta.

### `EvdbError evdb_sync_export_to_file(s, path, since)`
Export delta to a file.

### `EvdbError evdb_sync_import_from_file(s, path)`
Import delta from a file.

### `uint64_t evdb_sync_current_clock(s)`
Get current sync clock value.

## Utilities

### `const char* evdb_version_string(void)` → `"1.0.0"`
### `const char* evdb_error_string(EvdbError err)`
### `void evdb_set_log_level(int level)` → 0=off, 1=error, 2=info, 3=debug

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `EVDB_OK` | Success |
| 1 | `EVDB_ERR_NULL_HANDLE` | Null handle passed |
| 2 | `EVDB_ERR_IO` | I/O or internal error |
| 3 | `EVDB_ERR_OOM` | Out of memory |
| 4 | `EVDB_ERR_NOT_FOUND` | Record not found |
| 5 | `EVDB_ERR_INVALID_ARG` | Invalid argument |
| 6 | `EVDB_ERR_ONNX` | ONNX Runtime error |
| 7 | `EVDB_ERR_SYNC` | Sync error |

## Config Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `storage_dir` | required | Directory for `.bin` files |
| `hnsw_M` | 16 | HNSW max connections per node |
| `hnsw_ef_construction` | 200 | Build-time search width |
| `hnsw_ef_search` | 64 | Query-time search width |
| `ranker_alpha` | 0.70 | Cosine similarity weight |
| `ranker_beta` | 0.20 | Page proximity weight |
| `ranker_gamma` | 0.10 | Keyword match weight |
| `token_budget` | 3200 | Max tokens in RAG context |
| `embedding_threads` | 2 | ONNX inference threads |
| `enable_knowledge_graph` | 1 | Auto-extract entities |
| `enable_sync` | 0 | CRDT sync engine |
