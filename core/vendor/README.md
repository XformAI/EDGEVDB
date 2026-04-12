# EdgeVDB Vendored Dependencies

> **Third-party libraries vendored for reproducible, zero-dependency builds.**

This directory contains vendored third-party dependencies that EdgeVDB requires for building. By vendoring these libraries, we ensure:

- **Reproducible builds** — No internet access required at build time
- **Zero external dependencies** — Self-contained source tree
- **Version control** — Pinned versions for stability
- **Cross-platform** — Works offline on any platform
- **License compliance** — All dependencies are permissively licensed

## Contents

### nlohmann/json.hpp

**Single-header JSON library for modern C++**

- **Version**: 3.11.3 (vendored)
- **Source**: https://github.com/nlohmann/json
- **License**: MIT
- **Purpose**: JSON serialization/deserialization for object store and sync protocol
- **Usage**: Included via `#include <nlohmann/json.hpp>`

**Key Features:**
- Intuitive syntax to manipulate JSON data
- Integration with STL containers
- JSON Schema validation
- Serialization/deserialization of user-defined types
- No external dependencies, single header

**Example Usage:**
```cpp
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Create JSON object
json j = {
    {"name", "EdgeVDB"},
    {"version", "1.0.0"},
    {"features", {"hnsw", "kg", "sync"}}
};

// Serialize
std::string s = j.dump();

// Deserialize
json j2 = json::parse(s);
```

**Why Vendored:**
- Single-header library, easy to vendor
- Widely used and stable API
- Zero dependencies
- MIT license compatible with EdgeVDB

### doctest/doctest.h

**Single-header C++ testing framework**

- **Version**: 2.4.11 (vendored)
- **Source**: https://github.com/doctest/doctest
- **License**: MIT
- **Purpose**: Unit testing framework for C++ core
- **Usage**: Included via `#include "doctest.h"`

**Key Features:**
- Lightest C++ testing framework (single header)
- Fast compilation
- Rich assertion macros
- Test cases and subcases
- Benchmarking support
- No external dependencies

**Example Usage:**
```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("HNSW index operations") {
    HNSWIndex index(16, 200, 64);
    
    CHECK(index.size() == 0);
    
    float embedding[384] = {0};
    index.insert(1, embedding);
    
    CHECK(index.size() == 1);
}
```

**Why Vendored:**
- Single-header library, easy to vendor
- Minimal overhead
- No external dependencies
- MIT license compatible with EdgeVDB
- Faster than Catch2 for rapid iteration

### onnxruntime/onnxruntime_c_api.h

**ONNX Runtime C API header**

- **Version**: 1.20.0 (header vendored)
- **Source**: https://github.com/microsoft/onnxruntime
- **License**: MIT
- **Purpose**: ONNX model inference for embedding generation
- **Usage**: Included via `#include "onnxruntime_c_api.h"`

**Important Notes:**
- **Only the header is vendored** in this repository
- **Platform-specific shared libraries must be obtained separately:**
  - Linux: `libonnxruntime.so`
  - macOS: `libonnxruntime.dylib`
  - Windows: `onnxruntime.dll`
  - Android: `libonnxruntime.so` (via Maven Central)
  - iOS: `onnxruntime.framework`
- These libraries are **NOT committed to git** due to size
- Link against the appropriate library for your platform

**Obtaining ONNX Runtime Libraries:**

**Desktop (Linux/macOS/Windows):**
```bash
# Download pre-built binaries
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar xzf onnxruntime-linux-x64-1.20.0.tgz
cp onnxruntime-linux-x64-1.20.0/lib/*.so core/vendor/onnxruntime/
```

**Android:**
```kotlin
// Add to build.gradle.kts
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
```

**iOS:**
```bash
# Download iOS framework
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-ios-1.20.0.tgz
tar xzf onnxruntime-ios-1.20.0.tgz
```

**Why Header Only Vendored:**
- Shared libraries are platform-specific and large
- Header provides ABI stability
- Libraries can be obtained from official releases
- MIT license compatible with EdgeVDB

## Version Policy

### Vendored Library Updates

When updating vendored libraries:

1. **Check compatibility** — Ensure new version is API-compatible
2. **Review changes** — Check changelog for breaking changes
3. **Run tests** — Ensure all tests pass with new version
4. **Update documentation** — Note version change in this README
5. **Commit changes** — Include version number in commit message

### Current Versions

| Library | Version | Last Updated |
|---------|---------|--------------|
| nlohmann/json | 3.11.3 | 2024-01-15 |
| doctest | 2.4.11 | 2024-01-15 |
| onnxruntime (header) | 1.20.0 | 2024-01-15 |

## Adding New Dependencies

### Guidelines

Before vendoring a new dependency:

1. **Evaluate necessity** — Can the feature be implemented without it?
2. **Check license** — Must be compatible with Apache 2.0 (MIT, BSD, Apache, etc.)
3. **Assess size** — Prefer small, single-header libraries
4. **Review stability** — Choose mature, well-maintained libraries
5. **Check dependencies** — Prefer zero-dependency libraries

### Process

1. Download the library source
2. Place in appropriate subdirectory
3. Update this README with:
   - Library name and version
   - Source URL
   - License information
   - Purpose and usage
   - Why vendored
4. Add to `.gitignore` if large binaries are needed
5. Update `CMakeLists.txt` if necessary

## License Compliance

All vendored libraries are licensed under permissive licenses compatible with EdgeVDB's Apache 2.0 license:

| Library | License | Compatibility |
|---------|---------|---------------|
| nlohmann/json | MIT | ✅ Compatible |
| doctest | MIT | ✅ Compatible |
| onnxruntime | MIT | ✅ Compatible |

**License Files:**
- Each vendored library includes its own license file in its directory
- EdgeVDB's license file (`../../LICENSE`) includes attribution notices

## Build Integration

### CMake Integration

All vendored headers are automatically included via the CMake build system:

```cmake
# core/CMakeLists.txt
add_library(edgevdb_vendor INTERFACE)
target_include_directories(edgevdb_vendor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)
target_link_libraries(edgevdb_core PUBLIC edgevdb_vendor)
```

### Usage in Source Code

```cpp
// Include vendored headers
#include <nlohmann/json.hpp>
#include "doctest.h"
#include "onnxruntime_c_api.h"

// Use directly
json j = /* ... */;
CHECK(j["version"] == "1.0.0");
// ONNX Runtime API calls...
```

## Troubleshooting

### Missing ONNX Runtime Libraries

If you get linker errors for ONNX Runtime:

1. Download the appropriate library for your platform
2. Place it in `core/vendor/onnxruntime/`
3. Add to your linker flags: `-Lpath/to/onnxruntime -lonnxruntime`
4. Or use the platform-specific package manager

### Header Not Found

If you get "file not found" errors:

1. Ensure you're building from the repository root
2. Check that `CMAKE_CURRENT_SOURCE_DIR/vendor` is correct
3. Verify the file exists in the vendor directory
4. Check include paths in your build configuration

### Version Mismatch

If you encounter API compatibility issues:

1. Check the version in this README
2. Verify the vendored file version matches
3. Update to the version specified in this README
4. Run tests to ensure compatibility

## Security

### Vulnerability Management

Vendored dependencies are monitored for security vulnerabilities:

- Regular checks against CVE databases
- Prompt updates for critical vulnerabilities
- Security advisories in repository issues

### Reporting Vulnerabilities

If you discover a security vulnerability in a vendored dependency:

1. Check the upstream project for existing reports
2. Report to the upstream project maintainers
3. Open an issue in the EdgeVDB repository
4. Include the CVE ID (if available) and severity assessment

## See Also

- [../../README.md](../../README.md) — Project overview
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
- [../../LICENSE](../../LICENSE) — EdgeVDB license
