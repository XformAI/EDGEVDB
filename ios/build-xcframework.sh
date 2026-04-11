#!/bin/bash
# Build EdgeVDB xcframework for iOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "Building EdgeVDB for iOS..."

# Build for device (arm64)
cmake -S "${SCRIPT_DIR}/.." -B "${BUILD_DIR}/ios-arm64" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DEDGEVDB_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}/ios-arm64" --config Release

# Build for simulator (arm64)
cmake -S "${SCRIPT_DIR}/.." -B "${BUILD_DIR}/ios-sim-arm64" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DEDGEVDB_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}/ios-sim-arm64" --config Release

echo "Creating xcframework..."
xcodebuild -create-xcframework \
    -library "${BUILD_DIR}/ios-arm64/core/libedgevdb_core.a" \
    -headers "${SCRIPT_DIR}/../core/include" \
    -library "${BUILD_DIR}/ios-sim-arm64/core/libedgevdb_core.a" \
    -headers "${SCRIPT_DIR}/../core/include" \
    -output "${BUILD_DIR}/EdgeVDB.xcframework"

echo "Done! EdgeVDB.xcframework is at ${BUILD_DIR}/EdgeVDB.xcframework"
