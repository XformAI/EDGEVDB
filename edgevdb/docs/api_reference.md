# EdgeVDB C API Reference

## Header
```c
#include "edgevdb/vectordb.h"
```

## Lifecycle

### `evdb_default_config(EvdbConfig* out)`
Fill a config struct with default values.

### `EvdbHandle* evdb_open(const EvdbConfig* config)`
Open or create a database. Returns NULL on failure.

### `void evdb_close(EvdbHandle* h)`
Save and close a database handle.

### `EvdbError evdb_save(EvdbHandle* h)`
Flush all data to disk.

## Embedding

### `EvdbEmbedder* evdb_embedder_create(model_path, vocab_path, threads)`
Create an embedder from ONNX model and vocab files.

### `void evdb_embedder_destroy(EvdbEmbedder* e)`
Free an embedder.

### `EvdbError evdb_embed_text(EvdbEmbedder* e, text, float* out_384)`
Produce a 384-dim L2-normalized embedding.

## Vector Store

### `EvdbError evdb_insert_text(h, embedder, text, doc_id, page, &chunk_id)`
Embed + insert text. Returns chunk ID.

### `EvdbError evdb_insert_chunk(h, text, embedding, doc_id, page, &chunk_id)`
Insert with pre-computed embedding.

### `EvdbError evdb_remove_chunk(h, chunk_id)`
Remove a chunk.

## Query

### `EvdbQueryHandle* evdb_query_text(h, embedder, query, top_k, use_kg)`
Query with auto-embedding.

### `EvdbQueryHandle* evdb_query_vector(h, embedding, text, top_k)`
Query with pre-computed embedding.

### Result accessors:
- `evdb_result_count(q)` → int
- `evdb_result_text(q, i)` → char*
- `evdb_result_score(q, i)` → float
- `evdb_result_chunk_id(q, i)` → uint64
- `evdb_result_page(q, i)` → uint32
- `evdb_result_context_string(q)` → char* (assembled RAG context)

### `void evdb_query_free(q)`
Free query results.

## Object Store

### `EvdbError evdb_object_put(h, type_name, json_properties, &id)`
Insert/update an object.

### `EvdbError evdb_object_get(h, id, out_json, size)`
Get object as JSON string.

### `EvdbError evdb_object_remove(h, id)`
Soft-delete an object.

### `EvdbError evdb_object_query(h, type, prop, value, out, size, limit)`
Query objects by type/property.

## Relations

### `EvdbError evdb_relation_add(h, name, from_id, to_id)`
Add a typed relation.

### `EvdbError evdb_relation_get_targets(h, name, from_id, out_ids, &count)`
Get target IDs for a relation.

## Sync

### `EvdbSyncEngine* evdb_sync_create(h, device_id)`
Create a sync engine.

### `EvdbError evdb_sync_export_delta(s, since, out, size)`
Export changes since a clock value.

### `EvdbError evdb_sync_apply_delta(s, json, &applied, &skipped)`
Apply a received delta.

## Utilities

### `const char* evdb_version_string(void)` → "1.0.0"
### `const char* evdb_error_string(EvdbError err)`
### `void evdb_set_log_level(int level)` → 0=off, 1=error, 2=info, 3=debug

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | EVDB_OK | Success |
| 1 | EVDB_ERR_NULL_HANDLE | Null handle |
| 2 | EVDB_ERR_IO | I/O error |
| 3 | EVDB_ERR_OOM | Out of memory |
| 4 | EVDB_ERR_NOT_FOUND | Not found |
| 5 | EVDB_ERR_INVALID_ARG | Invalid argument |
| 6 | EVDB_ERR_ONNX | ONNX Runtime error |
| 7 | EVDB_ERR_SYNC | Sync error |
