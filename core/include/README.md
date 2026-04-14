# EdgeVDB Core — Public API Headers

> **Stable C API and C++ convenience headers for the EdgeVDB vector database core.**

This directory contains the public API headers that external consumers include to use EdgeVDB. All headers are designed for:

- **ABI stability** — Binary compatibility across versions
- **C compatibility** — Usable from C, C++, and other languages via FFI
- **Zero dependencies** — Self-contained, no external includes required
- **Cross-platform** — Works on Linux, macOS, Windows, Android, iOS, Raspberry Pi

## Header Files

### `edgevdb/vectordb.h`

The **primary public API** — a stable C interface using opaque handle types. This is the ONLY header external consumers need to include for C or cross-language integration.

**Key Design Principles:**
- Opaque handle types (`EvdbHandle*`, `EvdbQueryHandle*`, etc.) hide implementation
- C linkage (`extern "C"`) ensures ABI stability
- No C++ features, pure C99 compatible
- Error codes instead of exceptions for C compatibility

**Handle Types:**
```c
typedef struct EvdbHandle_       EvdbHandle;        // Main database handle
typedef struct EvdbQueryHandle_  EvdbQueryHandle;   // Query result handle
typedef struct EvdbEmbedder_     EvdbEmbedder;      // Embedder handle
typedef struct EvdbSyncEngine_   EvdbSyncEngine;    // Sync engine handle
```

**Error Codes:**
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

**Configuration:**
```c
typedef struct {
    const char* storage_dir;         // Directory for .bin files
    int hnsw_M;                      // HNSW M parameter, default 16
    int hnsw_ef_construction;        // HNSW ef_construction, default 200
    int hnsw_ef_search;              // HNSW ef_search, default 64
    float ranker_alpha;              // Cosine weight, default 0.70
    float ranker_beta;               // Page proximity weight, default 0.20
    float ranker_gamma;              // Keyword weight, default 0.10
    int token_budget;                // Max tokens in context, default 3200
    int embedding_threads;           // ONNX thread count, default 2
    int enable_knowledge_graph;      // 0 or 1, default 1
    int enable_sync;                 // 0 or 1, default 0
    const char* device_id;           // For sync, NULL = auto-generate
} EvdbConfig;
```

**API Categories:**

1. **Lifecycle**
   - `evdb_open()` — Open database
   - `evdb_close()` — Close and free
   - `evdb_save()` — Save to disk
   - `evdb_default_config()` — Fill config with defaults

2. **Embedding Pipeline**
   - `evdb_embedder_create()` — Create embedder from ONNX model
   - `evdb_embedder_destroy()` — Free embedder
   - `evdb_embed_text()` — Embed text to 384-dim vector

3. **Vector Store — Insert**
   - `evdb_insert_text()` — Insert with auto-embedding
   - `evdb_insert_chunk()` — Insert with pre-computed embedding
   - `evdb_remove_chunk()` — Remove by ID

4. **Vector Store — Query**
   - `evdb_query_text()` — Query with auto-embedding
   - `evdb_query_vector()` — Query with pre-computed embedding
   - `evdb_result_count/text/score/chunk_id/page/context_string()` — Access results
   - `evdb_query_free()` — Free query handle

5. **Object Store**
   - `evdb_object_put()` — Store JSON object
   - `evdb_object_get()` — Retrieve by ID
   - `evdb_object_remove()` — Soft delete
   - `evdb_object_query()` — Query by type and property

6. **Relations**
   - `evdb_relation_add()` — Add typed edge
   - `evdb_relation_get_targets()` — Get target IDs

7. **Sync**
   - `evdb_sync_create()` — Create sync engine
   - `evdb_sync_destroy()` — Free sync engine
   - `evdb_sync_export_delta()` — Export changes since clock
   - `evdb_sync_apply_delta()` — Apply received delta
   - `evdb_sync_export_to_file()` — Export to file
   - `evdb_sync_import_from_file()` — Import from file
   - `evdb_sync_current_clock()` — Get current clock value

8. **Utilities**
   - `evdb_version_string()` — Get version
   - `evdb_error_string()` — Get error description
   - `evdb_set_log_level()` — Set verbosity (0=off, 1=error, 2=info, 3=debug)

### `edgevdb/edgevdb.hpp`

Optional C++ RAII wrapper providing idiomatic C++ convenience classes on top of the C API.

**Purpose:**
- Automatic resource management via RAII
- Exception-based error handling
- Type-safe wrappers around C handles
- Convenient for pure C++ applications

**Classes:**

1. **`EdgeVDBError`** — Exception class wrapping `EvdbError`
   ```cpp
   class EdgeVDBError : public std::runtime_error {
   public:
       EvdbError code;
       EdgeVDBError(EvdbError c, const std::string& msg);
   };
   ```

2. **`QueryResultsRAII`** — RAII wrapper for `EvdbQueryHandle`
   - Auto-frees handle in destructor
   - Provides convenient accessors: `count()`, `text()`, `score()`, `chunkId()`, `page()`, `contextString()`

3. **`EmbedderRAII`** — RAII wrapper for `EvdbEmbedder`
   - Auto-destroys embedder in destructor
   - `valid()` check for successful creation

4. **`DatabaseRAII`** — RAII wrapper for `EvdbHandle`
   - Auto-closes database in destructor
   - `save()` method with exception handling
   - `valid()` check for successful opening

**Usage Example:**
```cpp
#include "edgevdb/edgevdb.hpp"

using namespace edgevdb;

int main() {
    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = "./mydb";

    DatabaseRAII db(config);
    if (!db.valid()) {
        std::cerr << "Failed to open database\n";
        return 1;
    }

    // Use db.get() for C API calls, or extend wrapper
    db.save();
    return 0;
}
```

### `edgevdb/version.h`

Version information header.

**Contents:**
- Version string constant
- Version number components (major, minor, patch)
- Build metadata

**Usage:**
```cpp
#include "edgevdb/version.h"

std::cout << "EdgeVDB version: " << EDGEVDB_VERSION_STRING << std::endl;
```

## Integration Guide

### C Integration

```c
#include "edgevdb/vectordb.h"

int main() {
    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = "./data";

    EvdbHandle* db = evdb_open(&config);
    if (!db) return 1;

    // Use database...
    evdb_save(db);
    evdb_close(db);
    return 0;
}
```

### C++ Integration (with RAII wrapper)

```cpp
#include "edgevdb/edgevdb.hpp"

int main() {
    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = "./data";

    DatabaseRAII db(config);
    if (!db.valid()) return 1;

    // Use database...
    db.save();
    return 0;  // Auto-closes via destructor
}
```

### C++ Integration (direct C API)

```cpp
#include "edgevdb/vectordb.h"
#include <stdexcept>

int main() {
    EvdbConfig config;
    evdb_default_config(&config);
    config.storage_dir = "./data";

    EvdbHandle* db = evdb_open(&config);
    if (!db) throw std::runtime_error("Failed to open");

    // Use database...
    evdb_save(db);
    evdb_close(db);
    return 0;
}
```

### Cross-Language Integration (FFI)

The C API is designed for easy FFI bindings:

**Python (ctypes):**
```python
import ctypes

lib = ctypes.CDLL("libedgevdb.so")

lib.evdb_open.argtypes = [ctypes.c_void_p]
lib.evdb_open.restype = ctypes.c_void_p

# ... set up signatures for other functions
```

**Go (cgo):**
```go
/*
#cgo LDFLAGS: -ledgevdb
#include "edgevdb/vectordb.h"
*/
import "C"

func OpenDB(path string) unsafe.Pointer {
    cPath := C.CString(path)
    defer C.free(unsafe.Pointer(cPath))
    
    var config C.EvdbConfig
    C.evdb_default_config(&config)
    config.storage_dir = cPath
    
    return C.evdb_open(&config)
}
```

**Rust (bindgen):**
```rust
// Use bindgen to generate bindings from vectordb.h
include!("edgevdb_bindings.rs");

fn main() {
    let mut config = EvdbConfig::default();
    config.storage_dir = std::ptr::null();
    
    let db = unsafe { evdb_open(&config) };
    // ...
}
```

## ABI Stability

EdgeVDB guarantees ABI stability for the C API across minor versions (e.g., 1.0.x → 1.1.x). This means:

- **Binary compatibility**: Applications linked against `libedgevdb.so` v1.0.0 will work with v0.1.1 without recompilation
- **Handle opaqueness**: Handle sizes and layouts never change
- **Function signatures**: No function signatures change (only additions)
- **Error codes**: Error code values are stable

**Version Policy:**
- **Major version (X.0.0)**: Breaking changes possible
- **Minor version (1.X.0)**: Additions only, ABI stable
- **Patch version (1.0.X)**: Bug fixes only, ABI stable

## Platform-Specific Notes

### Android

- Headers are compatible with NDK r25+
- Use via JNI bindings (see `android/sdk/src/main/cpp/vectordb_jni.cpp`)
- ARM NEON SIMD automatically enabled for arm64-v8a

### iOS

- Headers work with Objective-C++ (.mm files)
- Swift bridging via module map (see `ios/Sources/EdgeVDBC/`)
- ARM NEON SIMD automatically enabled for arm64

### Desktop (Linux/macOS/Windows)

- Standard C/C++ compilation
- SSE2 SIMD automatically enabled for x86_64
- No special flags required

### Raspberry Pi

- ARM NEON SIMD automatically enabled for armv7/aarch64
- Tested on Raspberry Pi 4 and 5

## Memory Management

**Rules:**
1. All handles returned from `*_create` or `*_open` must be freed with corresponding `*_destroy` or `*_close`
2. Query result handles must be freed with `evdb_query_free()`
3. String pointers returned from the API are valid until the next API call on the same handle
4. The C++ RAII wrapper handles cleanup automatically

**Example:**
```c
EvdbHandle* db = evdb_open(&config);
EvdbQueryHandle* results = evdb_query_vector(db, embedding, query, 5);
// Use results...
evdb_query_free(results);  // Must free before closing db
evdb_close(db);
```

## Thread Safety

- **Database handle**: Thread-safe for concurrent reads, single writer
- **Query handles**: Not thread-safe, use one per thread
- **Embedder handles**: Thread-safe for concurrent `evdb_embed_text()` calls
- **Sync handles**: Thread-safe with external locking
- **Configuration**: Read-only after `evdb_open()`, thread-safe

## Best Practices

1. **Always check return values**: NULL handles indicate failure
2. **Use `evdb_error_string()`** for human-readable error messages
3. **Close handles in reverse order**: Close query handles before database handles
4. **Set log level during development**: `evdb_set_log_level(3)` for debug output
5. **Use RAII wrappers in C++**: Automatic resource management
6. **Batch operations**: Minimize API calls for better performance
7. **Pre-compute embeddings**: When possible, avoid ONNX overhead

## Troubleshooting

### Link Errors

Ensure you're linking against the correct library:
- Static: `libedgevdb_core.a`
- Shared: `libedgevdb_shared.so` (Linux), `libedgevdb_shared.dylib` (macOS), `edgevdb_shared.dll` (Windows)

### Runtime Errors

- **EVDB_ERR_NULL_HANDLE**: Check that you're passing valid handles
- **EVDB_ERR_IO**: Check file permissions and disk space
- **EVDB_ERR_OOM**: Reduce HNSW parameters or chunk size
- **EVDB_ERR_ONNX**: Verify ONNX model and vocab file paths

### ABI Mismatch

Ensure header version matches library version:
```c
printf("Library version: %s\n", evdb_version_string());
```

## See Also

- [../../README.md](../../README.md) — Project overview
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
- [../src/README.md](../src/README.md) — Implementation details
- [../../docs/api_reference.md](../../docs/api_reference.md) — Complete API documentation
