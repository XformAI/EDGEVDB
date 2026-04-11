#!/usr/bin/env python3
"""
quantize_model.py

Quantizes an ONNX model from FP32 to INT8 for mobile deployment.
Reduces model size by ~4× with <1% accuracy loss.

Usage:
    python3 quantize_model.py --input models/all-MiniLM-L6-v2.onnx \
                              --output models/model_quantized.onnx
"""

import argparse
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(input_path: str, output_path: str) -> None:
    """
    Dynamically quantize an ONNX model to INT8.

    Args:
        input_path:  Path to the input FP32 ONNX model.
        output_path: Path where the quantized model will be saved.
    """
    print(f"Loading model from {input_path}...")
    model = onnx.load(input_path)

    print(f"Quantizing to INT8...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,  # Use unsigned int8 for weights
        optimize_model=True
    )

    print(f"Quantized model saved to {output_path}")

    # Print size comparison
    import os
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100
    print(f"\nSize comparison:")
    print(f"  Input (FP32):  {input_size:.2f} MB")
    print(f"  Output (INT8): {output_size:.2f} MB")
    print(f"  Reduction:     {reduction:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input FP32 ONNX model"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save quantized INT8 ONNX model"
    )
    args = parser.parse_args()

    quantize_model(args.input, args.output)
