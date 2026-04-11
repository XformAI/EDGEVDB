#!/bin/bash
# Download all-MiniLM-L6-v2 model and vocab.
# See README.md in this directory for details.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading vocab.txt..."
curl -L "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" \
     -o "${SCRIPT_DIR}/vocab.txt"

echo ""
echo "vocab.txt downloaded. To get the ONNX model, run:"
echo "  pip install transformers onnx onnxruntime"
echo "  python -m transformers.onnx --model=sentence-transformers/all-MiniLM-L6-v2 --feature=default ${SCRIPT_DIR}/"
echo ""
echo "Then quantize (optional):"
echo "  python ../tools/convert_onnx.py ${SCRIPT_DIR}/model.onnx ${SCRIPT_DIR}/model-quantized.onnx"
