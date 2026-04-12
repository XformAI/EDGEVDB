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
endif()
