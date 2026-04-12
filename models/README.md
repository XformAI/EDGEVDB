# EdgeVDB Models

> **ONNX models and vocabulary files for on-device text embedding generation.**

This directory contains the ONNX model files and vocabulary required for EdgeVDB's built-in embedding functionality. The models enable on-device text-to-vector conversion without external API calls.

## Model Information

### all-MiniLM-L6-v2

**Sentence-BERT model optimized for semantic similarity**

- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture**: BERT-based (6 layers, 384 hidden dimensions)
- **Output**: 384-dimensional embeddings
- **Max Sequence Length**: 512 tokens
- **Vocabulary**: WordPiece with 30,522 tokens
- **License**: Apache 2.0

**Performance:**
- STS Benchmark: ~0.81 Spearman correlation
- Embedding Latency: ~20-50ms (desktop), ~50-100ms (mobile)
- Model Size: ~90 MB (FP32), ~22 MB (INT8 quantized)

## Required Files

| File | Description | Size (approx) | Required |
|------|-------------|---------------|----------|
| `model.onnx` | ONNX model file (FP32 or INT8) | 22-90 MB | Yes |
| `vocab.txt` | WordPiece vocabulary file | ~230 KB | Yes |

### Model Variants

#### FP32 Model (Full Precision)
- **File**: `all-MiniLM-L6-v2.onnx`
- **Size**: ~90 MB
- **Accuracy**: Best quality
- **Use Case**: Desktop applications, servers
- **Download**: Export from Hugging Face (see below)

#### INT8 Quantized Model
- **File**: `all-MiniLM-L6-v2-quantized.onnx`
- **Size**: ~22 MB
- **Accuracy**: ~0.1% loss on STS benchmarks
- **Use Case**: Mobile devices, edge deployment
- **Download**: Quantize from FP32 model (see below)

## Download Instructions

### Option 1: Automated Download Script

Use the provided download script to fetch the vocabulary file:

```bash
cd tools
python download_model.py
```

This downloads:
- `vocab.txt` from Hugging Face
- Places it in the `../models/` directory

**Note:** The ONNX model must be exported manually (see Option 2).

### Option 2: Manual Download and Export

#### Step 1: Install Dependencies

```bash
pip install transformers onnx onnxruntime
```

#### Step 2: Export to ONNX

```bash
cd models

# Export the model to ONNX format
python -m transformers.onnx \
    --model=sentence-transformers/all-MiniLM-L6-v2 \
    --feature=default \
    .
```

This creates `model.onnx` in the current directory.

#### Step 3: Download Vocabulary

```bash
# Download vocab.txt from Hugging Face
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
```

Or use the download script:
```bash
cd ../tools
python download_model.py
```

## Quantization

### Why Quantize?

Quantization reduces model size and improves inference speed with minimal accuracy loss:

| Metric | FP32 | INT8 | Improvement |
|--------|------|------|-------------|
| Model Size | ~90 MB | ~22 MB | 4× smaller |
| Inference Speed | Baseline | 1.5-2× faster | 2× faster |
| Accuracy (STS) | 0.81 | 0.809 | ~0.1% loss |

### Quantization Process

#### Step 1: Install ONNX Runtime with Quantization Support

```bash
pip install onnxruntime onnxruntime-tools
```

#### Step 2: Quantize the Model

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize to INT8
quantize_dynamic(
    input_model="model.onnx",
    output_model="model-quantized.onnx",
    weight_type=QuantType.QInt8,
    optimize_model=True
)

print("Quantization complete!")
```

Or use the provided script:

```bash
cd models
python ../tools/convert_onnx.py
```

#### Step 3: Verify Quantized Model

```python
import onnxruntime as ort

# Load and check the model
session = ort.InferenceSession("model-quantized.onnx")
print("Input names:", session.get_inputs()[0].name)
print("Output names:", session.get_outputs()[0].name)
print("Input shape:", session.get_inputs()[0].shape)
```

### Quantization Options

```python
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_preprocess

# Option 1: Dynamic quantization (recommended)
quantize_dynamic(
    "model.onnx",
    "model-quantized.onnx",
    weight_type=QuantType.QInt8
)

# Option 2: Static quantization (requires calibration data)
quantize_preprocess(
    input_model="model.onnx",
    output_model="model-static-quantized.onnx",
    calibration_data_path="./calibration_data"
)
```

## Model Verification

### Test the Model

```python
import onnxruntime as ort
import numpy as np

# Load the model
session = ort.InferenceSession("model.onnx")

# Prepare input (dummy data)
input_ids = np.random.randint(0, 30522, (1, 128), dtype=np.int64)
attention_mask = np.ones((1, 128), dtype=np.int64)
token_type_ids = np.zeros((1, 128), dtype=np.int64)

# Run inference
outputs = session.run(
    None,
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }
)

print("Output shape:", outputs[0].shape)
print("Output dtype:", outputs[0].dtype)
```

### Expected Output Shape

- **Token-level**: `(1, sequence_length, 384)` — Requires mean pooling
- **Pooled**: `(1, 384)` — Ready to use

EdgeVDB's embedder handles both shapes automatically.

## Usage in EdgeVDB

### C++

```cpp
#include "edgevdb/vectordb.h"

EvdbConfig config;
evdb_default_config(&config);
config.storage_dir = "./data";

EvdbHandle* db = evdb_open(&config);

// Create embedder
EvdbEmbedder* embedder = evdb_embedder_create(
    "models/model.onnx",  // Path to ONNX model
    "models/vocab.txt",   // Path to vocabulary
    2                      // Number of threads
);

// Embed text
float embedding[384];
evdb_embed_text(embedder, "Hello world", embedding);

// Clean up
evdb_embedder_destroy(embedder);
evdb_close(db);
```

### Python

```python
from edgevdb import EdgeVDB, Embedder

# Create embedder
embedder = Embedder(
    model_path="models/model.onnx",
    vocab_path="models/vocab.txt",
    threads=2
)

# Open database
with EdgeVDB("./data") as db:
    # Embed and insert
    chunk_id = db.insert_text(embedder, "Hello world", doc_id=1, page_number=0)
    
    # Query
    results = db.query_text(embedder, "greeting", top_k=5)
    print(results.context_string)
```

### Android (Kotlin)

```kotlin
// Copy model and vocab to app/src/main/assets/
// model.onnx
// vocab.txt

val embedder = Embedder.fromAssets(
    context = this,
    modelAsset = "model.onnx",
    vocabAsset = "vocab.txt",
    threads = 2
)

val embedding = embedder.embed("Hello world")
```

### iOS (Swift)

```swift
// Copy model and vocab to app bundle
// model.onnx
// vocab.txt

let embedder = try Embedder(
    modelURL: Bundle.main.url(forResource: "model", withExtension: "onnx")!,
    vocabURL: Bundle.main.url(forResource: "vocab", withExtension: "txt")!,
    threads: 2
)

let embedding = try embedder.embed("Hello world")
```

## Model Architecture Details

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Hidden Size | 384 |
| Attention Heads | 12 |
| Intermediate Size | 1536 |
| Max Position Embeddings | 512 |
| Vocabulary Size | 30,522 |
| Tokenizer Type | WordPiece |
| Output Embedding Size | 384 |

### Input Format

**Inputs:**
- `input_ids`: `[batch_size, sequence_length]` (int64)
- `attention_mask`: `[batch_size, sequence_length]` (int64)
- `token_type_ids`: `[batch_size, sequence_length]` (int64)

**Output:**
- `last_hidden_state`: `[batch_size, sequence_length, 384]` (float32)
- `pooler_output`: `[batch_size, 384]` (float32) — optional

### Tokenization

**Special Tokens:**
- `[CLS]`: ID 101 — Classification token
- `[SEP]`: ID 102 — Separator token
- `[PAD]`: ID 0 — Padding token
- `[UNK]`: ID 100 — Unknown token

**Max Sequence Length:** 512 tokens
**Recommended Chunk Size:** 200-300 words (for RAG applications)

## Troubleshooting

### Model Loading Errors

**Error:** `Failed to load model`
- **Solution:** Verify the model file path is correct
- **Check:** File exists and is readable
- **Verify:** File is a valid ONNX model

**Error:** `Vocabulary not found`
- **Solution:** Verify vocab.txt path is correct
- **Check:** File contains 30,522 lines
- **Verify:** No corrupted lines or missing tokens

### Out of Memory Errors

**Error:** `Out of memory during inference`
- **Solution:** Use quantized model
- **Reduce:** Batch size (if batching)
- **Decrease:** Max sequence length

### Slow Inference

**Issue:** Embedding takes too long
- **Solution:** Use quantized INT8 model
- **Enable:** NNAPI delegate (Android) or CoreML (iOS)
- **Reduce:** Max sequence length
- **Increase:** Thread count (if CPU supports it)

## Model Alternatives

While `all-MiniLM-L6-v2` is the default and recommended model, you can use other sentence-transformers models:

### Alternative Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 90 MB | Fast | Good | Default, mobile |
| all-mpnet-base-v2 | 420 MB | Medium | Better | Desktop, server |
| paraphrase-MiniLM-L6-v2 | 80 MB | Fast | Good | Paraphrase detection |
| e5-small-v2 | 33 MB | Very Fast | Good | Low-resource devices |

**Note:** Alternative models may require:
- Different vocabulary files
- Different input/output shapes
- Adjustments to embedder code

## Security Considerations

### Model Integrity

- **Checksums**: Verify model integrity using SHA256
- **Source**: Download only from official Hugging Face
- **Validation**: Test model before deployment

### Privacy

- **On-device**: Models run locally, no data leaves device
- **No telemetry**: EdgeVDB does not collect usage data
- **Offline**: Works without internet connection

## Performance Benchmarks

### Desktop (Intel i7, 16GB RAM)

| Operation | FP32 | INT8 |
|-----------|------|------|
| Load Model | 200ms | 200ms |
| Embed (128 tokens) | 25ms | 15ms |
| Embed (512 tokens) | 80ms | 45ms |

### Mobile (Snapdragon 8 Gen 2)

| Operation | FP32 | INT8 |
|-----------|------|------|
| Load Model | 500ms | 500ms |
| Embed (128 tokens) | 60ms | 35ms |
| Embed (512 tokens) | 180ms | 100ms |

### Raspberry Pi 4

| Operation | FP32 | INT8 |
|-----------|------|------|
| Load Model | 1200ms | 1200ms |
| Embed (128 tokens) | 150ms | 90ms |
| Embed (512 tokens) | 450ms | 280ms |

## References

- **Model Paper**: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- **Hugging Face**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **ONNX Runtime**: https://onnxruntime.ai/
- **Transformers**: https://huggingface.co/docs/transformers

## See Also

- [../tools/download_model.py](../tools/download_model.py) — Download script
- [../tools/convert_onnx.py](../tools/convert_onnx.py) — Quantization script
- [../README.md](../README.md) — Project overview
- [../../DEVELOPER_GUIDE.md](../../DEVELOPER_GUIDE.md) — Build and integration guide
