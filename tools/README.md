# EdgeVDB Tools

> **Utility scripts and tools for EdgeVDB development and deployment.**

This directory contains helper scripts and tools for working with EdgeVDB, including model download, ONNX conversion, and index validation.

## Tools

### download_model.py

Downloads the vocabulary file for the all-MiniLM-L6-v2 embedding model from Hugging Face.

**Usage:**

```bash
python download_model.py
```

**What it does:**
- Downloads `vocab.txt` from Hugging Face
- Places it in the `../models/` directory
- Skips download if file already exists
- Provides instructions for ONNX model export

**Output:**
```
Downloading to /path/to/EDGEVDB/models...
  Downloading vocab.txt...
  ✓ vocab.txt (230,123 bytes)

Downloaded 1/1 files.

NOTE: ONNX model must be exported manually. Run:

    pip install transformers onnx onnxruntime
    python -m transformers.onnx \
        --model=sentence-transformers/all-MiniLM-L6-v2 \
        --feature=default \
        {models_dir}

Then optionally quantize:

    python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    '{models_dir}/model.onnx',
    '{models_dir}/all-MiniLM-L6-v2-quantized.onnx',
    weight_type=QuantType.QInt8
)
"
```

**Requirements:**
- Python 3.6+
- Standard library only (no external dependencies)

**Source:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### convert_onnx.py

Quantizes an ONNX model to INT8 for mobile deployment.

**Usage:**

```bash
python convert_onnx.py
```

**What it does:**
- Loads `model.onnx` from the models directory
- Quantizes it to INT8 using dynamic quantization
- Saves the quantized model as `model-quantized.onnx`
- Reports file size reduction

**Output:**
```
Loading model: model.onnx
Original size: 90,234,567 bytes
Quantizing...
Quantization complete!
Quantized size: 22,456,789 bytes
Size reduction: 75.1%
```

**Requirements:**
- Python 3.6+
- `onnxruntime` package
- `onnxruntime-tools` package (for quantization)

**Installation:**

```bash
pip install onnxruntime onnxruntime-tools
```

**Custom Usage:**

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    input_model="model.onnx",
    output_model="model-quantized.onnx",
    weight_type=QuantType.QInt8,
    optimize_model=True
)
```

### validate_index.cpp

C++ tool for validating HNSW index files and checking database integrity.

**Usage:**

```bash
# Build the tool first
cmake --preset desktop-debug
cmake --build build/desktop-debug

# Run the validator
./build/desktop-debug/tools/validate_index
```

**What it does:**
- Checks if HNSW index file exists
- Validates file format (magic bytes, version)
- Verifies CRC32 checksums
- Reports statistics (node count, dimensions, etc.)
- Checks for corrupted data

**Output:**

```
EdgeVDB Index Validator
=======================

Validating: hnsw.bin
✓ File exists
✓ Magic bytes match (EVDBHNSW)
✓ Version: 1
✓ CRC32 checksum valid
✓ Node count: 10000
✓ Dimensions: 384
✓ Entry point: 1234
✓ Max layer: 5

Index is valid.
```

**Error Output:**

```
EdgeVDB Index Validator
=======================

Validating: hnsw.bin
✗ File not found
```

**Command-line Options:**

```bash
# Validate specific index file
./validate_index --file path/to/hnsw.bin

# Verbose mode (detailed output)
./validate_index --verbose

# Check all index files in directory
./validate_index --directory ./mydb
```

**Building:**

The validator is built as part of the CMake build process. To build standalone:

```bash
cd tools
g++ -std=c++17 -I../core/include -I../core/vendor \
    validate_index.cpp -L../build/desktop-debug/core \
    -ledgevdb_shared -o validate_index
```

## Adding New Tools

### Python Scripts

1. Create the script in the `tools/` directory
2. Add shebang: `#!/usr/bin/env python3`
3. Add docstring with usage instructions
4. Update this README with tool description
5. Make script executable: `chmod +x script_name.py`

**Example:**

```python
#!/usr/bin/env python3
"""
Tool description here.

Usage:
    python my_tool.py [options]
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="My tool")
    parser.add_argument("--option", help="Option description")
    args = parser.parse_args()
    
    # Tool logic here
    print("Tool executed successfully")

if __name__ == "__main__":
    main()
```

### C++ Tools

1. Create the source file in the `tools/` directory
2. Include necessary headers from `../core/include/`
3. Link against `libedgevdb_shared`
4. Update this README with tool description
5. Add build instructions

**Example:**

```cpp
#include "edgevdb/hnsw_index.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Tool executed successfully" << std::endl;
    return 0;
}
```

## Common Workflows

### Setting Up Models for Development

```bash
# Download vocabulary
cd tools
python download_model.py

# Export ONNX model
pip install transformers onnx onnxruntime
cd ../models
python -m transformers.onnx \
    --model=sentence-transformers/all-MiniLM-L6-v2 \
    --feature=default \
    .

# Quantize for mobile
cd ../tools
python convert_onnx.py
```

### Validating Database After Migration

```bash
# Validate all index files in database
cd build/desktop-debug/tools
./validate_index --directory ../../my_database
```

### Batch Processing

```bash
# Download and convert in one script
cd tools
python download_model.py
python convert_onnx.py
```

## Troubleshooting

### download_model.py Issues

**Error:** `Failed to download vocab.txt`

**Solutions:**
- Check internet connection
- Verify Hugging Face is accessible
- Try downloading manually from the URL
- Check file permissions in models directory

**Error:** `Permission denied when writing file`

**Solutions:**
- Ensure models directory is writable
- Run with appropriate permissions
- Create models directory if it doesn't exist

### convert_onnx.py Issues

**Error:** `ModuleNotFoundError: No module named 'onnxruntime'`

**Solutions:**
- Install dependencies: `pip install onnxruntime onnxruntime-tools`
- Use a virtual environment to isolate dependencies
- Verify Python version compatibility

**Error:** `Failed to load model.onnx`

**Solutions:**
- Ensure model.onnx exists in models directory
- Verify file is a valid ONNX model
- Check file permissions
- Ensure model was exported correctly

### validate_index.cpp Issues

**Error:** `Failed to open index file`

**Solutions:**
- Verify file path is correct
- Check file exists
- Ensure file is readable
- Run with verbose mode for details

**Error:** `CRC32 checksum mismatch`

**Solutions:**
- Index file may be corrupted
- Restore from backup if available
- Rebuild the index from source data

## Contributing

When adding new tools:

1. **Documentation**: Add clear usage instructions in the script/docstring
2. **Error Handling**: Include meaningful error messages
3. **Testing**: Test the tool on multiple platforms
4. **README**: Update this README with tool description
5. **Dependencies**: Minimize external dependencies when possible

## See Also

- [../models/README.md](../models/README.md) — Model documentation
- [../README.md](../README.md) — Project overview
- [../DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) — Build and integration guide
