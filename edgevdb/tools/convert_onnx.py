#!/usr/bin/env python3
"""
ONNX model quantization helper for EdgeVDB.

Converts full-precision ONNX model to INT8 dynamic quantization.
Reduces model size by ~4x with minimal accuracy loss.

Usage:
    python convert_onnx.py input.onnx output-quantized.onnx
"""

import sys
import os


def quantize_model(input_path: str, output_path: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print(f"Input:  {input_path} ({os.path.getsize(input_path) / 1e6:.1f} MB)")

        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )

        print(f"Output: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
        ratio = os.path.getsize(output_path) / os.path.getsize(input_path)
        print(f"Compression: {ratio:.1%}")

    except ImportError:
        print("Install onnxruntime: pip install onnxruntime")
        sys.exit(1)


def validate_model(model_path: str):
    """Validate model produces 384-dim output."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(model_path)

        # Check input names
        input_names = [x.name for x in sess.get_inputs()]
        print(f"Inputs: {input_names}")

        # Dummy inference
        dummy_ids = np.zeros((1, 128), dtype=np.int64)
        dummy_mask = np.ones((1, 128), dtype=np.int64)
        dummy_types = np.zeros((1, 128), dtype=np.int64)

        feeds = {}
        if "input_ids" in input_names:
            feeds["input_ids"] = dummy_ids
        if "attention_mask" in input_names:
            feeds["attention_mask"] = dummy_mask
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = dummy_types

        outputs = sess.run(None, feeds)
        print(f"Output shape: {outputs[0].shape}")

        if outputs[0].shape[-1] == 384:
            print("✓ Model produces 384-dim embeddings")
        else:
            print(f"⚠ Expected 384-dim, got {outputs[0].shape[-1]}-dim")

    except Exception as e:
        print(f"Validation error: {e}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output-quantized.onnx>")
        print(f"       {sys.argv[0]} --validate <model.onnx>")
        sys.exit(1)

    if sys.argv[1] == "--validate":
        validate_model(sys.argv[2])
    else:
        quantize_model(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
