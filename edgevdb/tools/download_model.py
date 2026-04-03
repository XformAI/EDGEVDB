#!/usr/bin/env python3
"""
Download all-MiniLM-L6-v2 ONNX model and vocab for EdgeVDB.
"""

import os
import sys
import urllib.request
import shutil

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

FILES = {
    "vocab.txt": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
}

ONNX_NOTE = """
NOTE: ONNX model must be exported manually. Run:

    pip install transformers onnx onnxruntime
    python -m transformers.onnx \\
        --model=sentence-transformers/all-MiniLM-L6-v2 \\
        --feature=default \\
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
"""

def download_file(url, dest):
    print(f"  Downloading {os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"  ✓ {os.path.basename(dest)} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading to {MODELS_DIR}...")

    success = 0
    for filename, url in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"  Skip {filename} (already exists)")
            success += 1
        elif download_file(url, dest):
            success += 1

    print(f"\nDownloaded {success}/{len(FILES)} files.")
    print(ONNX_NOTE.format(models_dir=MODELS_DIR))

if __name__ == "__main__":
    main()
