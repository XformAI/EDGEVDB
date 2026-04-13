# EdgeVDB C API Reference

> **Complete reference for the EdgeVDB stable C API.**
> 
> Include: `#include "edgevdb/vectordb.h"`

## Overview

The EdgeVDB C API provides a stable ABI for cross-language integration. All functions use opaque handles to hide implementation details and ensure binary compatibility across versions.

**Key Design Principles:**
- **Opaque Handles**: All resources are accessed through opaque pointer types
- **Error Codes**: Functions return `EvdbError` for error handling
- **C Linkage**: `extern "C"` ensures ABI stability
- **Zero Dependencies**: No external library dependencies required

## Type Definitions

### Handle Types

```c
typedef struct EvdbHandle_       EvdbHandle;        // Main database handle
typedef struct EvdbQueryHandle_  EvdbQueryHandle;   // Query result handle
typedef struct EvdbEmbedder_     EvdbEmbedder;      // Embedder handle
typedef struct EvdbSyncEngine_   EvdbSyncEngine;    // Sync engine handle
```

### Configuration

```c
typedef struct {
    const char* storage_dir;         // Directory for .bin files (required)
    int hnsw_M;                      // HNSW M parameter (default: 16)
    int hnsw_ef_construction;        // HNSW ef_construction (default: 200)
    int hnsw_ef_search;              // HNSW ef_search (default: 64)
    float ranker_alpha;              // Cosine weight (default: 0.70)
    float ranker_beta;               // Page proximity weight (default: 0.20)
    float ranker_gamma;              // Keyword weight (default: 0.10)
    int token_budget;                // Max tokens in context (default: 3200)
    int embedding_threads;           // ONNX thread count (default: 2)
    int enable_knowledge_graph;      // 0 or 1, default 1
    int enable_sync;                 // 0 or 1, default 0
    const char* device_id;           // For sync, NULL = auto-generate
} EvdbConfig;
```

### Error Codes

```c
typedef enum {
    EVDB_OK              = 0,  // Success
    EVDB_ERR_NULL_HANDLE = 1,  // Null handle passed
    EVDB_ERR_IO          = 2,  // I/O or internal error
    EVDB_ERR_OOM         = 3,  // Out of memory
    EVDB_ERR_NOT_FOUND   = 4,  // Record not found
    EVDB_ERR_INVALID_ARG = 5,  // Invalid argument
    EVDB_ERR_ONNX        = 6,  // ONNX Runtime error
    EVDB_ERR_SYNC        = 7,  // Sync error
} EvdbError;
```

## Lifecycle Functions

### `void evdb_default_config(EvdbConfig* out)`

Fill a config struct with default values.

**Parameters:**
- `out`: Pointer to config struct to fill

**Example:**
```c
EvdbConfig config;
evdb_default_config(&config);
config.storage_dir = "./my_database";
config.hnsw_ef_search = 128;  // Override default
```

### `EvdbHandle* evdb_open(const EvdbConfig* config)`

Open or create a database. Returns `NULL` on failure.

**Parameters:**
- `config`: Pointer to configuration struct

**Returns:**
- Database handle on success
- `NULL` on failure

**Example:**
```c
EvdbConfig config;
evdb_default_config(&config);
config.storage_dir = "./my_database";

EvdbHandle* db = evdb_open(&config);
if (!db) {
    fprintf(stderr, "Failed to open database: %s\n", evdb_error_string(EVDB_ERR_IO));
    return 1;
}
```

### `void evdb_close(EvdbHandle* h)`

Close and free a database handle. Automatically calls `evdb_save()` before closing.

**Parameters:**
- `h`: Database handle

**Example:**
```c
evdb_save(db);  // Save before closing
evdb_close(db);
```

### `EvdbError evdb_save(EvdbHandle* h)`

Flush all data to disk.

**Parameters:**
- `h`: Database handle

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
EvdbError err = evdb_save(db);
if (err != EVDB_OK) {
    fprintf(stderr, "Save failed: %s\n", evdb_error_string(err));
}
```

## Embedding Functions (Optional)

> The embedder is optional. Use `insert_chunk()` and `query_vector()` with pre-computed embeddings to avoid any ONNX dependency.

### `EvdbEmbedder* evdb_embedder_create(const char* model_path, const char* vocab_path, int threads)`

Create an embedder from ONNX model and vocab files. Returns `NULL` on failure.

**Parameters:**
- `model_path`: Path to ONNX model file (.onnx)
- `vocab_path`: Path to vocabulary file (vocab.txt)
- `threads`: Number of inference threads (default: 2)

**Returns:**
- Embedder handle on success
- `NULL` on failure

**Example:**
```c
EvdbEmbedder* embedder = evdb_embedder_create("model.onnx", "vocab.txt", 2);
if (!embedder) {
    fprintf(stderr, "Failed to create embedder\n");
    return 1;
}
```

### `void evdb_embedder_destroy(EvdbEmbedder* e)`

Free an embedder.

**Parameters:**
- `e`: Embedder handle

**Example:**
```c
evdb_embedder_destroy(embedder);
```

### `EvdbError evdb_embed_text(EvdbEmbedder* e, const char* text, float* out_384)`

Produce a 384-dim L2-normalized embedding.

**Parameters:**
- `e`: Embedder handle
- `text`: Input text to embed
- `out_384`: Output array (must be 384 floats)

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
float embedding[384];
EvdbError err = evdb_embed_text(embedder, "Hello world", embedding);
if (err != EVDB_OK) {
    fprintf(stderr, "Embedding failed: %s\n", evdb_error_string(err));
}
```

## Vector Store — Insert

### `EvdbError evdb_insert_text(EvdbHandle* h, EvdbEmbedder* embedder, const char* text, uint32_t doc_id, uint32_t page, uint64_t* out_chunk_id)`

Auto-embed text and insert. Requires an embedder.

**Parameters:**
- `h`: Database handle
- `embedder`: Embedder handle
- `text`: Text to insert
- `doc_id`: Document identifier
- `page`: Page number within document
- `out_chunk_id`: Output parameter for assigned chunk ID

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
uint64_t chunk_id;
EvdbError err = evdb_insert_text(db, embedder, "Neural networks classify images", 1, 0, &chunk_id);
if (err == EVDB_OK) {
    printf("Inserted chunk: %llu\n", chunk_id);
}
```

### `EvdbError evdb_insert_chunk(EvdbHandle* h, const char* text, const float* embedding_384, uint32_t doc_id, uint32_t page, uint64_t* out_chunk_id)`

Insert with a pre-computed 384-dim embedding. **No embedder needed.**

**Parameters:**
- `h`: Database handle
- `text`: Text to insert
- `embedding_384`: Pre-computed embedding (384 floats)
- `doc_id`: Document identifier
- `page`: Page number within document
- `out_chunk_id`: Output parameter for assigned chunk ID

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
float embedding[384];
// ... compute embedding ...

uint64_t chunk_id;
EvdbError err = evdb_insert_chunk(db, "Your text here", embedding, 1, 0, &chunk_id);
```

### `EvdbError evdb_remove_chunk(EvdbHandle* h, uint64_t chunk_id)`

Remove a chunk by ID.

**Parameters:**
- `h`: Database handle
- `chunk_id`: Chunk ID to remove

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
EvdbError err = evdb_remove_chunk(db, chunk_id);
```

## Vector Store — Query

### `EvdbQueryHandle* evdb_query_text(EvdbHandle* h, EvdbEmbedder* embedder, const char* query, int top_k, int use_kg)`

Query with auto-embedding. Requires an embedder.

**Parameters:**
- `h`: Database handle
- `embedder`: Embedder handle
- `query`: Query text
- `top_k`: Maximum number of results
- `use_kg`: Whether to use knowledge graph expansion (0 or 1)

**Returns:**
- Query result handle on success
- `NULL` on failure

**Example:**
```c
EvdbQueryHandle* results = evdb_query_text(db, embedder, "image classification", 5, 0);
if (results) {
    // Process results
    evdb_query_free(results);
}
```

### `EvdbQueryHandle* evdb_query_vector(EvdbHandle* h, const float* embedding_384, const char* query_text_for_keyword, int top_k)`

Query with a pre-computed embedding. **No embedder needed.**

**Parameters:**
- `h`: Database handle
- `embedding_384`: Pre-computed query embedding (384 floats)
- `query_text_for_keyword`: Query text for keyword scoring (can be NULL)
- `top_k`: Maximum number of results

**Returns:**
- Query result handle on success
- `NULL` on failure

**Example:**
```c
float query_embedding[384];
// ... compute embedding ...

EvdbQueryHandle* results = evdb_query_vector(db, query_embedding, "image classification", 5);
if (results) {
    // Process results
    evdb_query_free(results);
}
```

### Result Accessors

| Function | Returns | Description |
|----------|---------|-------------|
| `evdb_result_count(EvdbQueryHandle* q)` | `int` | Number of results |
| `evdb_result_text(EvdbQueryHandle* q, int i)` | `const char*` | Text at index |
| `evdb_result_score(EvdbQueryHandle* q, int i)` | `float` | Score at index |
| `evdb_result_chunk_id(EvdbQueryHandle* q, int i)` | `uint64_t` | Chunk ID at index |
| `evdb_result_page(EvdbQueryHandle* q, int i)` | `uint32_t` | Page number at index |
| `evdb_result_context_string(EvdbQueryHandle* q)` | `const char*` | Assembled RAG context |

**Example:**
```c
int count = evdb_result_count(results);
for (int i = 0; i < count; i++) {
    const char* text = evdb_result_text(results, i);
    float score = evdb_result_score(results, i);
    uint64_t chunk_id = evdb_result_chunk_id(results, i);
    uint32_t page = evdb_result_page(results, i);
    
    printf("[%llu] [page %u] [%.3f] %s\n", chunk_id, page, score, text);
}

const char* context = evdb_result_context_string(results);
printf("Context:\n%s\n", context);
```

### `void evdb_query_free(EvdbQueryHandle* q)`

Free query results. **Always call this after processing results.**

**Parameters:**
- `q`: Query result handle

**Example:**
```c
evdb_query_free(results);
```

## Object Store

### `EvdbError evdb_object_put(EvdbHandle* h, const char* type_name, const char* json_properties, uint64_t* out_id)`

Insert or update an object. Returns object ID.

**Parameters:**
- `h`: Database handle
- `type_name`: Object type (e.g., "Document")
- `json_properties`: JSON string with object properties
- `out_id`: Output parameter for object ID

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
uint64_t obj_id;
const char* json = "{\"title\":\"ML Guide\",\"author\":\"Alice\"}";
EvdbError err = evdb_object_put(db, "Document", json, &obj_id);
```

### `EvdbError evdb_object_get(EvdbHandle* h, uint64_t id, char* out_json, int out_json_size)`

Get object as JSON string.

**Parameters:**
- `h`: Database handle
- `id`: Object ID
- `out_json`: Output buffer for JSON string
- `size`: Size of output buffer

**Returns:**
- `EVDB_OK` on success
- `EVDB_ERR_NOT_FOUND` if object not found
- Error code on failure

**Example:**
```c
char json[4096];
EvdbError err = evdb_object_get(db, obj_id, json, sizeof(json));
if (err == EVDB_OK) {
    printf("Object: %s\n", json);
}
```

### `EvdbError evdb_object_remove(EvdbHandle* h, uint64_t id)`

Soft-delete an object.

**Parameters:**
- `h`: Database handle
- `id`: Object ID

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
EvdbError err = evdb_object_remove(db, obj_id);
```

### `EvdbError evdb_object_query(EvdbHandle* h, const char* type_name, const char* filter_property, const char* filter_value, char* out_json_array, int out_size, int limit)`

Query objects by type and property filter.

**Parameters:**
- `h`: Database handle
- `type`: Object type to filter
- `prop`: Property name to filter
- `value`: Property value to match
- `out`: Output buffer for JSON array
- `size`: Size of output buffer
- `limit`: Maximum number of results

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
char out[4096];
EvdbError err = evdb_object_query(db, "Document", "author", "Alice", out, sizeof(out), 10);
```

## Relations

### `EvdbError evdb_relation_add(EvdbHandle* h, const char* name, uint64_t from_id, uint64_t to_id)`

Add a typed directional relation.

**Parameters:**
- `h`: Database handle
- `name`: Relation type (e.g., "has_chunk")
- `from_id`: Source object ID
- `to_id`: Target object ID

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
EvdbError err = evdb_relation_add(db, "has_chunk", doc_id, chunk_id);
```

### `EvdbError evdb_relation_get_targets(EvdbHandle* h, const char* name, uint64_t from_id, uint64_t* out_ids, int* count)`

Get target IDs for a relation from a source. `count` is an **in-out** parameter: caller sets `*count` to the buffer capacity of `out_ids`; on return, `*count` is set to the actual number of targets written.

**Parameters:**
- `h`: Database handle
- `name`: Relation type
- `from_id`: Source object ID
- `out_ids`: Output buffer for target IDs
- `count`: In-out parameter for buffer capacity and result count

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
uint64_t targets[100];
int count = 100;  // Buffer capacity
EvdbError err = evdb_relation_get_targets(db, "has_chunk", doc_id, targets, &count);
if (err == EVDB_OK) {
    for (int i = 0; i < count; i++) {
        printf("Target: %llu\n", targets[i]);
    }
}
```

## Sync

### `EvdbSyncEngine* evdb_sync_create(EvdbHandle* h, const char* device_id)`

Create a sync engine.

**Parameters:**
- `h`: Database handle
- `device_id`: Device identifier (NULL for auto-generated)

**Returns:**
- Sync engine handle on success
- `NULL` on failure

**Example:**
```c
EvdbSyncEngine* sync = evdb_sync_create(db, "device-123");
if (!sync) {
    fprintf(stderr, "Failed to create sync engine\n");
}
```

### `void evdb_sync_destroy(EvdbSyncEngine* s)`

Destroy a sync engine.

**Parameters:**
- `s`: Sync engine handle

**Example:**
```c
evdb_sync_destroy(sync);
```

### `EvdbError evdb_sync_export_delta(EvdbSyncEngine* s, uint64_t since, char* out, size_t size)`

Export changes since a clock value as JSON.

**Parameters:**
- `s`: Sync engine handle
- `since`: Clock value to export from
- `out`: Output buffer for JSON delta
- `size`: Size of output buffer

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
char delta[65536];
EvdbError err = evdb_sync_export_delta(sync, 0, delta, sizeof(delta));
```

### `EvdbError evdb_sync_apply_delta(EvdbSyncEngine* s, const char* json, int* applied, int* skipped)`

Apply a received delta.

**Parameters:**
- `s`: Sync engine handle
- `json`: JSON delta string
- `applied`: Output parameter for number of applied records
- `skipped`: Output parameter for number of skipped records

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
int applied, skipped;
EvdbError err = evdb_sync_apply_delta(sync, delta_json, &applied, &skipped);
printf("Applied: %d, Skipped: %d\n", applied, skipped);
```

### `EvdbError evdb_sync_export_to_file(EvdbSyncEngine* s, const char* path, uint64_t since)`

Export delta to a file.

**Parameters:**
- `s`: Sync engine handle
- `path`: File path to write delta
- `since`: Clock value to export from

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
EvdbError err = evdb_sync_export_to_file(sync, "delta.json", 0);
```

### `EvdbError evdb_sync_import_from_file(EvdbSyncEngine* s, const char* path)`

Import delta from a file.

**Parameters:**
- `s`: Sync engine handle
- `path`: File path to read delta

**Returns:**
- `EVDB_OK` on success
- Error code on failure

**Example:**
```c
EvdbError err = evdb_sync_import_from_file(sync, "delta.json");
```

### `uint64_t evdb_sync_current_clock(EvdbSyncEngine* s)`

Get current sync clock value.

**Parameters:**
- `s`: Sync engine handle

**Returns:**
- Current clock value

**Example:**
```c
uint64_t clock = evdb_sync_current_clock(sync);
printf("Current clock: %llu\n", clock);
```

## Utilities

### `const char* evdb_version_string(void)`

Get version string.

**Returns:**
- Version string (e.g., "1.0.0")

**Example:**
```c
printf("EdgeVDB version: %s\n", evdb_version_string());
```

### `const char* evdb_error_string(EvdbError err)`

Get human-readable error description.

**Parameters:**
- `err`: Error code

**Returns:**
- Error description string

**Example:**
```c
fprintf(stderr, "Error: %s\n", evdb_error_string(err));
```

### `void evdb_set_log_level(int level)`

Set logging verbosity.

**Parameters:**
- `level`: Log level (0=off, 1=error, 2=info, 3=debug)

**Example:**
```c
evdb_set_log_level(3);  // Enable debug logging
```

## Complete Example

```c
#include "edgevdb/vectordb.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Configure database
    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = "./my_database";
    
    // Open database
    EvdbHandle* db = evdb_open(&config);
    if (!db) {
        fprintf(stderr, "Failed to open database\n");
        return 1;
    }
    
    // Create embedder (optional)
    EvdbEmbedder* embedder = evdb_embedder_create("model.onnx", "vocab.txt", 2);
    if (!embedder) {
        fprintf(stderr, "Failed to create embedder\n");
        evdb_close(db);
        return 1;
    }
    
    // Insert text with auto-embedding
    uint64_t chunk_id;
    EvdbError err = evdb_insert_text(db, embedder, "Machine learning finds patterns", 1, 0, &chunk_id);
    if (err != EVDB_OK) {
        fprintf(stderr, "Insert failed: %s\n", evdb_error_string(err));
        evdb_embedder_destroy(embedder);
        evdb_close(db);
        return 1;
    }
    
    // Query with auto-embedding
    EvdbQueryHandle* results = evdb_query_text(db, embedder, "what is ML?", 5, 0);
    if (results) {
        int count = evdb_result_count(results);
        for (int i = 0; i < count; i++) {
            const char* text = evdb_result_text(results, i);
            float score = evdb_result_score(results, i);
            printf("[%.3f] %s\n", score, text);
        }
        evdb_query_free(results);
    }
    
    // Save and cleanup
    evdb_save(db);
    evdb_embedder_destroy(embedder);
    evdb_close(db);
    
    return 0;
}
```

## See Also

- [architecture.md](architecture.md) — Architecture overview
- [../core/include/README.md](../core/include/README.md) — Public API headers
- [../README.md](../README.md) — Project overview
