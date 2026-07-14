# tests/CMakeLists.txt — EdgeVDB unit and integration tests

# Threads are not available on iOS in the same way as Unix systems
find_package(Threads)

# Helper: create a test from a source file
function(edgevdb_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE edgevdb_core)
    if(TARGET Threads::Threads)
        target_link_libraries(${TEST_NAME} PRIVATE Threads::Threads)
    endif()
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/core/vendor
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Unit tests
edgevdb_add_test(test_hnsw test_hnsw.cpp)
edgevdb_add_test(test_hybrid_ranker test_hybrid_ranker.cpp)
edgevdb_add_test(test_embedder test_embedder.cpp)
edgevdb_add_test(test_object_store test_object_store.cpp)
edgevdb_add_test(test_sync test_sync.cpp)
edgevdb_add_test(test_e2e_rag test_e2e_rag.cpp)
edgevdb_add_test(test_crc test_crc.cpp)
edgevdb_add_test(test_tokenizer test_tokenizer.cpp)
edgevdb_add_test(test_persistence test_persistence.cpp)
edgevdb_add_test(test_concurrency test_concurrency.cpp)
edgevdb_add_test(test_sync_convergence test_sync_convergence.cpp)
edgevdb_add_test(test_ab_recall test_ab_recall.cpp)
edgevdb_add_test(test_onnx_semantic test_onnx_semantic.cpp)
edgevdb_add_test(test_bm25 test_bm25.cpp)

# Benchmarks — only in Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_executable(bench_query benchmarks/bench_query_latency.cpp)
    target_link_libraries(bench_query PRIVATE edgevdb_core)
    if(TARGET Threads::Threads)
        target_link_libraries(bench_query PRIVATE Threads::Threads)
    endif()
    target_include_directories(bench_query PRIVATE
        ${CMAKE_SOURCE_DIR}/core/vendor
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_SOURCE_DIR}/core/include
    )

    add_executable(bench_build benchmarks/bench_build_time.cpp)
    target_link_libraries(bench_build PRIVATE edgevdb_core)
    if(TARGET Threads::Threads)
        target_link_libraries(bench_build PRIVATE Threads::Threads)
    endif()
    target_include_directories(bench_build PRIVATE
        ${CMAKE_SOURCE_DIR}/core/vendor
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_SOURCE_DIR}/core/include
    )

    add_executable(bench_scale benchmarks/bench_scale.cpp)
    target_link_libraries(bench_scale PRIVATE edgevdb_core)
    if(TARGET Threads::Threads)
        target_link_libraries(bench_scale PRIVATE Threads::Threads)
    endif()
    target_include_directories(bench_scale PRIVATE
        ${CMAKE_SOURCE_DIR}/core/vendor
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_SOURCE_DIR}/core/include
    )

    add_executable(bench_extended benchmarks/bench_extended.cpp)
    target_link_libraries(bench_extended PRIVATE edgevdb_core)
    if(TARGET Threads::Threads)
        target_link_libraries(bench_extended PRIVATE Threads::Threads)
    endif()
    target_include_directories(bench_extended PRIVATE
        ${CMAKE_SOURCE_DIR}/core/vendor
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_SOURCE_DIR}/core/include
    )
endif()
