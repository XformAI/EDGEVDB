cmake_minimum_required(VERSION 3.22)
project(edgevdb_android VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Compiler flags ──────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wpedantic
    -fvisibility=hidden          # hide all symbols by default
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_compile_options(-march=armv8-a+simd -DEDGEVDB_NEON=1)
elseif(ANDROID_ABI STREQUAL "x86_64")
    add_compile_options(-msse4.2 -DEDGEVDB_SSE2=1)
endif()

# ── Vendor header-only libraries ────────────────────────────────────
add_library(vendor_headers INTERFACE)
target_include_directories(vendor_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/vendor
)

# ── EdgeVDB core static library ─────────────────────────────────────
set(EDGEVDB_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../core)

file(GLOB EDGEVDB_SOURCES "${EDGEVDB_CORE_DIR}/src/*.cpp")
# Exclude test-only files if any
list(FILTER EDGEVDB_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")

add_library(edgevdb_core STATIC ${EDGEVDB_SOURCES})
target_include_directories(edgevdb_core PUBLIC
    ${EDGEVDB_CORE_DIR}/include
    ${EDGEVDB_CORE_DIR}/src
)
target_link_libraries(edgevdb_core PUBLIC vendor_headers)

# ── JNI shared library ───────────────────────────────────────────────
add_library(edgevdb_jni SHARED
    src/main/cpp/vectordb_jni.cpp
)

target_include_directories(edgevdb_jni PRIVATE
    ${EDGEVDB_CORE_DIR}/include
    ${EDGEVDB_CORE_DIR}/src
)

target_link_libraries(edgevdb_jni
    edgevdb_core
    android          # Android log / asset APIs
    log
)

# Strip debug symbols in release to minimize .so size
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET edgevdb_jni POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-unneeded
        $<TARGET_FILE:edgevdb_jni>
    )
endif()
