# EdgeVDB Vendored Dependencies

This directory contains vendored third-party dependencies for reproducible builds
with zero internet access at build time.

## Contents

### nlohmann/json.hpp
- Single-header JSON library for C++
- Source: https://github.com/nlohmann/json
- License: MIT

### onnxruntime/onnxruntime_c_api.h
- ONNX Runtime C API header
- Source: https://github.com/microsoft/onnxruntime
- License: MIT
- **Note:** Platform-specific shared libraries (.so/.dylib/.dll) must be placed
  in this directory for each target platform. These are NOT committed to git.

### doctest/doctest.h
- Single-header C++ testing framework
- Source: https://github.com/doctest/doctest
- License: MIT
