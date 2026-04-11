# Assets Directory

This directory must contain the ONNX model and vocabulary files for the embedding pipeline.

## Required Files

1. **model.onnx** - The quantized all-MiniLM-L6-v2 ONNX model (~22 MB)
2. **vocab.txt** - BERT vocabulary file (~230 KB, 30,522 lines)

## How to Obtain These Files

### Option 1: Use the existing EdgeVDB models directory

If you have the EdgeVDB repository with the `edgevdb/models/` directory:

```bash
cp ../../edgevdb/models/all-MiniLM-L6-v2-quantized.onnx model.onnx
cp ../../edgevdb/models/vocab.txt vocab.txt
```

### Option 2: Export and quantize from Hugging Face

1. **Export to ONNX:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install optimum transformers torch onnxruntime

python3 -c "
from optimum.exporters.onnx import main_export
main_export('sentence-transformers/all-MiniLM-L6-v2', './models', task='feature-extraction')
"
```

2. **Quantize the model:**
```bash
python3 ../../quantize_model.py --input models/model.onnx --output model.onnx
```

3. **Copy vocab.txt:**
The vocab.txt is typically included in the exported model directory.

4. **Copy to assets:**
```bash
cp model.onnx edgevdb/android/src/main/assets/model.onnx
cp vocab.txt edgevdb/android/src/main/assets/vocab.txt
```

## Important Notes

- The model file is large and should NOT be committed to Git
- Add these lines to your `.gitignore`:
```
edgevdb/android/src/main/assets/model.onnx
edgevdb/android/src/main/assets/vocab.txt
```
- For production builds, ensure the assets are properly bundled in the APK/AAB
