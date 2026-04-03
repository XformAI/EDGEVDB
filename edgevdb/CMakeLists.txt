cmake_minimum_required(VERSION 3.22)
project(edgevdb VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EDGEVDB_BUILD_TESTS "Build tests" ON)
option(EDGEVDB_BUILD_PYTHON "Build Python extension" OFF)
option(EDGEVDB_ENABLE_NEON "Enable ARM NEON SIMD" ON)
option(EDGEVDB_ENABLE_SSE2 "Enable x86 SSE2 SIMD" ON)
option(EDGEVDB_ENABLE_SYNC "Build sync engine" ON)

add_subdirectory(core)

if(EDGEVDB_BUILD_TESTS AND NOT ANDROID)
  add_subdirectory(tests)
endif()
