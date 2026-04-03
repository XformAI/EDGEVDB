cmake_minimum_required(VERSION 3.22)

# Collect core sources
file(GLOB EDGEVDB_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../core/src/*.cpp")

add_library(edgevdb SHARED
    src/main/cpp/vectordb_jni.cpp
    ${EDGEVDB_CORE_SOURCES}
)

target_include_directories(edgevdb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/vendor
)

target_compile_options(edgevdb PRIVATE -O2 -fvisibility=hidden -DANDROID -DEDGEVDB_ANDROID)
target_compile_features(edgevdb PRIVATE cxx_std_17)

find_library(log-lib log)
find_library(android-lib android)
target_link_libraries(edgevdb ${log-lib} ${android-lib})
