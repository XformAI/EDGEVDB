# EdgeVDB Models

This directory is for storing ONNX model files and vocab files.

## Required Files

| File | Description | Size (approx) |
|------|-------------|---------------|
| `all-MiniLM-L6-v2.onnx` | Sentence-BERT embedding model | ~90 MB |
| `all-MiniLM-L6-v2-quantized.onnx` | INT8 quantized model | ~22 MB |
| `vocab.txt` | WordPiece vocabulary (30,522 tokens) | ~230 KB |

## Download

Use the download script:
```bash
cd tools
python download_model.py
```

Or download manually from Hugging Face:
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Export to ONNX using: `python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 models/`

## Quantization

For mobile deployment, quantize to INT8:
```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "all-MiniLM-L6-v2.onnx",
    "all-MiniLM-L6-v2-quantized.onnx",
    weight_type=QuantType.QInt8
)
```

The quantized model reduces size by ~4× with minimal accuracy loss (~0.1% on STS benchmarks).
