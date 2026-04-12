# EdgeVDB Tests

> **Comprehensive test suite for the EdgeVDB vector database core.**

This directory contains unit tests, integration tests, and benchmarks for the EdgeVDB C++ core library. The test suite ensures correctness, performance, and stability across all components.

## Test Structure

```
tests/
├── CMakeLists.txt              # Test build configuration
├── test_hnsw.cpp               # HNSW index unit tests
├── test_hybrid_ranker.cpp      # Hybrid ranking algorithm tests
├── test_embedder.cpp           # Embedding pipeline tests
├── test_object_store.cpp       # Object store CRUD tests
├── test_sync.cpp               # CRDT sync engine tests
├── test_e2e_rag.cpp            # End-to-end RAG pipeline tests
└── benchmarks/
    ├── bench_query_latency.cpp  # Query latency benchmarks
    └── bench_build_time.cpp     # Index build throughput benchmarks
```

## Running Tests

### Prerequisites

Build the EdgeVDB core library first:

```bash
# Debug build (includes tests)
cmake --preset desktop-debug
cmake --build build/desktop-debug

# Release build (includes benchmarks)
cmake --preset desktop-release
cmake --build build/desktop-release
```

### Run All Tests

```bash
# From project root
./build/desktop-debug/tests/test_hnsw
./build/desktop-debug/tests/test_hybrid_ranker
./build/desktop-debug/tests/test_embedder
./build/desktop-debug/tests/test_object_store
./build/desktop-debug/tests/test_sync
./build/desktop-debug/tests/test_e2e_rag
```

### Run Individual Tests

```bash
# HNSW index tests
./build/desktop-debug/tests/test_hnsw

# Hybrid ranker tests
./build/desktop-debug/tests/test_hybrid_ranker

# End-to-end RAG tests
./build/desktop-debug/tests/test_e2e_rag
```

### Run Benchmarks (Release Only)

```bash
# Query latency benchmark
./build/desktop-release/tests/bench_query

# Index build time benchmark
./build/desktop-release/tests/bench_build
```

### Run with CTest

```bash
cd build/desktop-debug
ctest --verbose
```

## Test Descriptions

### test_hnsw.cpp

**Unit tests for the HNSW ANN index.**

**Test Coverage:**
- Index creation and initialization
- Vector insertion
- k-NN search
- Index persistence (save/load)
- Deletion (soft delete)
- CRC32 checksum validation
- SIMD distance computation

**Key Test Cases:**
- Insert 1000 random vectors, verify all retrievable
- Search with different ef values, verify recall
- Save index to disk, load and verify integrity
- Delete vectors, verify they're not returned in search
- Test with different M parameters (8, 16, 32)

**Expected Output:**
```
[doctest] doctest version is "2.4.11"
[doctest] run with "--doctest-test-case=HNSW Index Operations"
===============================================================================
test cases: 15 | 15 passed | 0 failed
===============================================================================
```

### test_hybrid_ranker.cpp

**Unit tests for the hybrid ranking algorithm.**

**Test Coverage:**
- Cosine similarity scoring
- Page proximity scoring
- Keyword overlap scoring
- Weighted score combination
- Stopword removal
- Tokenization

**Key Test Cases:**
- Rank results with different weight combinations
- Verify keyword overlap with Jaccard coefficient
- Test page proximity calculation
- Ensure scores are in [0.0, 1.0] range
- Test with empty queries

**Expected Output:**
```
===============================================================================
test cases: 10 | 10 passed | 0 failed
===============================================================================
```

### test_embedder.cpp

**Unit tests for the embedding pipeline.**

**Test Coverage:**
- Embedder creation and destruction
- Text to embedding conversion
- ONNX model loading (if available)
- Fallback hash embedder
- L2 normalization
- Output dimension validation

**Key Test Cases:**
- Embed sample text, verify output is 384-dim
- Verify L2 normalization (norm ≈ 1.0)
- Test with empty text
- Test with very long text (truncation)
- Verify hash embedder produces consistent results

**Expected Output:**
```
===============================================================================
test cases: 8 | 8 passed | 0 failed
===============================================================================
```

### test_object_store.cpp

**Unit tests for the object store.**

**Test Coverage:**
- Object CRUD operations
- Type-based indexing
- Property-based indexing
- JSON serialization
- Persistence (save/load)
- Soft delete

**Key Test Cases:**
- Insert 100 objects, verify all retrievable
- Query by type, verify correct results
- Query by property value, verify filtering
- Save to disk, load and verify integrity
- Delete objects, verify they're not returned

**Expected Output:**
```
===============================================================================
test cases: 12 | 12 passed | 0 failed
===============================================================================
```

### test_sync.cpp

**Unit tests for the CRDT sync engine.**

**Test Coverage:**
- Sync engine creation
- Delta export/import
- LWW conflict resolution
- Vector clock management
- File-based sync

**Key Test Cases:**
- Create two databases, sync between them
- Export delta, apply to second database
- Verify LWW resolves conflicts correctly
- Test with concurrent writes
- Verify vector clock monotonicity

**Expected Output:**
```
===============================================================================
test cases: 6 | 6 passed | 0 failed
===============================================================================
```

### test_e2e_rag.cpp

**End-to-end RAG pipeline integration test.**

**Test Coverage:**
- Complete RAG workflow
- Document ingestion (chunking + embedding)
- Semantic search
- Context assembly
- Token budget management

**Key Test Cases:**
- Ingest a sample document
- Query the document
- Verify relevant chunks are returned
- Verify context string is assembled correctly
- Test with different token budgets

**Expected Output:**
```
===============================================================================
test cases: 5 | 5 passed | 0 failed
===============================================================================
```

## Benchmarks

### bench_query_latency.cpp

**Measures query latency at various scales.**

**Metrics:**
- Average query latency (ms)
- P50, P95, P99 latency
- Latency vs. database size
- Latency vs. ef parameter

**Running the Benchmark:**

```bash
./build/desktop-release/tests/bench_query
```

**Expected Output:**
```
Query Latency Benchmark
========================
Database size: 10000 vectors
ef_search: 64
Top-k: 10

Results:
- Average latency: 45.2 ms
- P50 latency: 42.1 ms
- P95 latency: 68.3 ms
- P99 latency: 89.7 ms
- QPS: 22.1
```

**Parameters:**
- Database size: 1K, 10K, 100K vectors
- ef_search: 32, 64, 128
- Top-k: 5, 10, 20

### bench_build_time.cpp

**Measures index build throughput.**

**Metrics:**
- Index build time (seconds)
- Insertion rate (vectors/second)
- Build time vs. database size
- Build time vs. M parameter

**Running the Benchmark:**

```bash
./build/desktop-release/tests/bench_build
```

**Expected Output:**
```
Index Build Time Benchmark
==========================
Database size: 10000 vectors
M: 16
ef_construction: 200

Results:
- Total build time: 2.3 s
- Insertion rate: 4347.8 vectors/s
- Average insert time: 0.23 ms
```

**Parameters:**
- Database size: 1K, 10K, 100K vectors
- M: 8, 16, 32
- ef_construction: 100, 200, 400

## Performance Targets

| Metric | Target | Platform |
|--------|--------|----------|
| Query latency (10k chunks) | < 100ms | Desktop |
| Query latency (10k chunks) | < 200ms | Raspberry Pi |
| Index build (10k chunks) | < 5s | Desktop |
| Index build (10k chunks) | < 10s | Raspberry Pi |
| Memory footprint (10k chunks) | < 100 MB | Desktop |

## Test Framework

EdgeVDB uses **doctest** as its testing framework:

- **Single-header**: No external dependencies
- **Lightweight**: Fast compilation
- **BDD-style**: Clean test syntax
- **Benchmarking**: Built-in support

### Writing New Tests

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "edgevdb/hnsw_index.hpp"

TEST_CASE("My new test") {
    // Arrange
    HNSWIndex index(16, 200, 64);
    
    // Act
    float embedding[384] = {0};
    index.insert(1, embedding);
    
    // Assert
    CHECK(index.size() == 1);
    
    // Cleanup (RAII handles this automatically)
}
```

### Test Organization

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Benchmarks**: Measure performance characteristics

### Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Use RAII for automatic resource cleanup
3. **Assertions**: Use descriptive assertion messages
4. **Edge Cases**: Test boundary conditions
5. **Performance**: Add benchmarks for critical paths

## Continuous Integration

Tests are run automatically on CI/CD:

```yaml
# Example CI configuration
steps:
  - name: Build
    run: |
      cmake --preset desktop-debug
      cmake --build build/desktop-debug
  
  - name: Test
    run: |
      cd build/desktop-debug
      ctest --verbose
  
  - name: Benchmark
    run: |
      cmake --preset desktop-release
      cmake --build build/desktop-release
      ./build/desktop-release/tests/bench_query
```

## Troubleshooting

### Test Failures

**Symptom:** Tests fail with segmentation fault

**Solutions:**
- Ensure the library was built in debug mode
- Check that the library version matches the test version
- Enable debug logging in the tests
- Run tests under a debugger (gdb/lldb)

**Symptom:** Tests fail with file not found

**Solutions:**
- Verify the test data directory exists
- Check file permissions
- Ensure tests are run from the correct directory

### Benchmark Failures

**Symptom:** Benchmarks show degraded performance

**Solutions:**
- Verify the build is in release mode
- Check that SIMD optimizations are enabled
- Close other applications consuming CPU
- Verify system is not under heavy load

## Adding New Tests

### Step 1: Create Test File

```cpp
// tests/test_my_feature.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "edgevdb/my_feature.hpp"

TEST_CASE("My feature test") {
    // Test implementation
}
```

### Step 2: Update CMakeLists.txt

```cmake
# tests/CMakeLists.txt
edgevdb_add_test(test_my_feature test_my_feature.cpp)
```

### Step 3: Build and Run

```bash
cmake --preset desktop-debug
cmake --build build/desktop-debug
./build/desktop-debug/tests/test_my_feature
```

### Step 4: Add to CI

Update CI configuration to run the new test.

## Coverage

To measure code coverage:

```bash
# Build with coverage flags
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage"
cmake --build build

# Run tests
./build/tests/test_hnsw
./build/tests/test_hybrid_ranker
# ... run all tests

# Generate coverage report
lcov --capture --directory build --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
```

## See Also

- [../README.md](../README.md) — Project overview
- [../DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) — Build and integration guide
- [../core/src/README.md](../core/src/README.md) — Implementation details
