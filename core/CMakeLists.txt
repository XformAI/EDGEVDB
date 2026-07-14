# core/CMakeLists.txt — EdgeVDB core C++ library

# Vendor include paths
add_library(edgevdb_vendor INTERFACE)
target_include_directories(edgevdb_vendor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)

# Collect all source files
set(EDGEVDB_CORE_SOURCES
    src/chunk_store.cpp
    src/page_index.cpp
    src/hnsw_index.cpp
    src/hybrid_ranker.cpp
    src/kg_extractor.cpp
    src/knowledge_graph.cpp
    src/kg_expander.cpp
    src/tokenizer.cpp
    src/embedder.cpp
    src/object_store.cpp
    src/relation_index.cpp
    src/query_engine.cpp
    src/sync_engine.cpp
    src/token_budget.cpp
    src/vectordb_impl.cpp
    src/vectordb_c_api.cpp
)

# Static library
add_library(edgevdb_core STATIC ${EDGEVDB_CORE_SOURCES})
target_include_directories(edgevdb_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(edgevdb_core PUBLIC edgevdb_vendor)
# The embedder dlopen()s ONNX Runtime at runtime; glibc < 2.34 keeps the
# dl* symbols in a separate libdl (CMAKE_DL_LIBS is empty on Windows).
target_link_libraries(edgevdb_core PUBLIC ${CMAKE_DL_LIBS})
target_compile_features(edgevdb_core PUBLIC cxx_std_17)

# Strict warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edgevdb_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Platform-specific
if(ANDROID)
    find_library(LOG_LIB log)
    find_library(ANDROID_LIB android)
    target_link_libraries(edgevdb_core PUBLIC ${LOG_LIB} ${ANDROID_LIB})
    target_compile_definitions(edgevdb_core PUBLIC EDGEVDB_ANDROID)
endif()

# SIMD options
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    target_compile_definitions(edgevdb_core PUBLIC EDGEVDB_ARM64)
    if(EDGEVDB_ENABLE_NEON AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(edgevdb_core PRIVATE -mfpu=neon)
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    if(EDGEVDB_ENABLE_SSE2)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(edgevdb_core PRIVATE -msse2)
        endif()
    endif()
endif()

# Shared library (same sources)
add_library(edgevdb_shared SHARED ${EDGEVDB_CORE_SOURCES})
target_include_directories(edgevdb_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(edgevdb_shared PUBLIC edgevdb_vendor)
target_link_libraries(edgevdb_shared PUBLIC ${CMAKE_DL_LIBS})
target_compile_features(edgevdb_shared PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edgevdb_shared PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# On MinGW, link the GCC runtime statically so the DLL is self-contained —
# consumers (e.g. Python ctypes) would otherwise fail to load it unless the
# MinGW bin directory is on PATH.
if(MINGW)
    target_link_options(edgevdb_shared PRIVATE -static-libgcc -static-libstdc++ -static)
endif()

# Platform-specific for shared library
if(ANDROID)
    target_link_libraries(edgevdb_shared PUBLIC ${LOG_LIB} ${ANDROID_LIB})
    target_compile_definitions(edgevdb_shared PUBLIC EDGEVDB_ANDROID)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    target_compile_definitions(edgevdb_shared PUBLIC EDGEVDB_ARM64)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    if(EDGEVDB_ENABLE_SSE2)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(edgevdb_shared PRIVATE -msse2)
        endif()
    endif()
endif()

# Install rules
install(TARGETS edgevdb_core edgevdb_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/edgevdb DESTINATION include)
