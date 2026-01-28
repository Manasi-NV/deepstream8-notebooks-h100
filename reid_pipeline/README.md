# Person Re-ID Pipeline for DeepStream 8.0

A complete guide to running Person Re-Identification as a Secondary GIE (SGIE) in DeepStream, extracting embedding vectors, and storing them in ChromaDB for gallery building.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup Guide](#setup-guide)
4. [Model Export (OSNet)](#model-export-osnet)
5. [Configuration](#configuration)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Common Errors & Solutions](#common-errors--solutions)

---

## Overview

### What is Person Re-ID?

Person Re-Identification (Re-ID) is the task of identifying the same person across different camera views or time frames. Unlike classification (which outputs labels like "car", "person"), Re-ID models output **embedding vectors** - numerical representations that can be compared for similarity.

### Key Differences from Classification SGIEs

| Aspect | Classification SGIE | Re-ID SGIE |
|--------|---------------------|------------|
| Output | Class labels (color: "red") | Embedding vector (512 floats) |
| Config | `is-classifier=1` | `output-tensor-meta=1` |
| Access | `classifier_meta_list` | `obj_user_meta_list` → `NvDsInferTensorMeta` |
| Use Case | Attribute recognition | Similarity matching |

---

## Architecture

### Pipeline Flow

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Source  │ → │ Decoder │ → │  PGIE   │ → │ Tracker │ → │Re-ID    │
│ (video) │   │         │   │(detect) │   │         │   │ SGIE    │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └────┬────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  Buffer Probe   │
                                                    │  (extract emb)  │
                                                    └────────┬────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │    ChromaDB     │
                                                    │   (storage)     │
                                                    └─────────────────┘
```

### Why This Order?

1. **PGIE first** → Detects persons in the frame
2. **Tracker second** → Assigns persistent `track_id` across frames
3. **Re-ID SGIE third** → Generates embeddings for tracked persons
4. **Probe after SGIE** → Extracts embeddings from tensor metadata

### Embedding Storage Strategy: Store on Track End

Instead of storing embeddings every frame (redundant), we:
1. Accumulate embeddings while a person is tracked
2. Wait for track to end (person leaves scene)
3. Average all embeddings for that person
4. Store single entry in ChromaDB

This gives **one gallery entry per unique person**.

---

## Setup Guide

### Prerequisites

- DeepStream 8.0 container with Python bindings
- NVIDIA GPU with TensorRT support
- Python 3.10+ (for model export, outside container)

### Step 1: Create Folder Structure

```bash
mkdir -p reid_pipeline/models/osnet reid_pipeline/gallery
```

### Step 2: Export OSNet Model

**Run OUTSIDE the DeepStream container** (needs PyTorch):

```bash
# Create virtual environment
python3 -m venv osnet_env
source osnet_env/bin/activate

# Install dependencies
pip install torchreid torch torchvision scipy opencv-python tensorboard onnx onnxscript

# Run export script
python export_osnet.py
```

### Step 3: Convert to Embedded Weights

The export creates two files - convert to single file:

```python
import onnx

model = onnx.load('osnet_x1_0.onnx', load_external_data=True)
onnx.save(model, 'osnet_x1_0_final.onnx', save_as_external_data=False)
```

### Step 4: Copy to Container

```bash
docker cp osnet_x1_0_final.onnx <container>:/app/notebooks/reid_pipeline/models/osnet/
docker cp reid_sgie_config.txt <container>:/app/notebooks/reid_pipeline/
docker cp person_reid_gallery_h100.ipynb <container>:/app/notebooks/reid_pipeline/
```

### Step 5: Build TensorRT Engine (Optional but Recommended)

Pre-building the engine saves time on first run:

```bash
docker exec <container> /usr/src/tensorrt/bin/trtexec \
  --onnx=/app/notebooks/reid_pipeline/models/osnet/osnet_x1_0_final.onnx \
  --saveEngine=/app/notebooks/reid_pipeline/models/osnet/osnet_x1_0.engine \
  --fp16
```

### Step 6: Install ChromaDB

```bash
docker exec <container> pip install chromadb
```

### Step 7: Run the Notebook

Open `person_reid_gallery_h100.ipynb` in Jupyter and run all cells.

---

## Model Export (OSNet)

### Why OSNet?

- Lightweight (~9MB)
- 512-dimensional embeddings
- Good accuracy on person Re-ID benchmarks
- Open source (torchreid library)

### Export Script (`export_osnet.py`)

```python
import torch
from torchreid import models
import onnx

# Load pretrained model
model = models.build_model(
    name='osnet_x1_0',
    num_classes=1,
    loss='softmax',
    pretrained=True
)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 256, 128)
torch.onnx.export(
    model,
    dummy_input,
    'osnet_x1_0.onnx',
    input_names=['input'],
    output_names=['embedding'],
    opset_version=17
)

# Convert to embedded weights (IMPORTANT!)
model = onnx.load('osnet_x1_0.onnx', load_external_data=True)
onnx.save(model, 'osnet_x1_0_final.onnx', save_as_external_data=False)
```

### Model Specifications

| Property | Value |
|----------|-------|
| Input Size | 3 x 256 x 128 (C x H x W) |
| Output Size | 512 (embedding dimension) |
| File Size | ~9 MB (with embedded weights) |
| Normalization | (pixel - 127.5) / 127.5 |

---

## Configuration

### SGIE Config (`reid_sgie_config.txt`)

```ini
[property]
gpu-id=0
gie-unique-id=4
process-mode=2

# Model paths - USE ABSOLUTE PATHS!
onnx-file=/app/notebooks/reid_pipeline/models/osnet/osnet_x1_0_final.onnx
model-engine-file=/app/notebooks/reid_pipeline/models/osnet/osnet_x1_0.engine

# CRITICAL: Output raw tensor, not classification
output-tensor-meta=1
network-type=100

# Only run on persons (class_id=2 from PGIE)
operate-on-gie-id=1
operate-on-class-ids=2

# Preprocessing (must match training)
net-scale-factor=0.0078431372549
offsets=127.5;127.5;127.5
model-color-format=0

# IMPORTANT: Must match engine batch size
batch-size=1

# FP16 for speed
network-mode=2

# Minimum person crop size
input-object-min-width=32
input-object-min-height=64
```

### Critical Config Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `output-tensor-meta` | 1 | Outputs raw tensor (embedding) instead of class labels |
| `network-type` | 100 | Custom network (not classifier/detector) |
| `batch-size` | 1 | Must match the TensorRT engine batch size |
| `operate-on-class-ids` | 2 | Only process persons (class 2) |

---

## Troubleshooting Guide

### Debugging Steps We Followed

This section documents the actual issues we encountered and how we solved them.

---

### Issue 1: Python Environment Error

**Symptom:**
```
error: externally-managed-environment
× This environment is externally managed
```

**Cause:** Modern Ubuntu/Debian prevents system-wide pip installs.

**Solution:**
```bash
python3 -m venv osnet_env
source osnet_env/bin/activate
pip install torchreid torch torchvision
```

---

### Issue 2: Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'scipy'
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'tensorboard'
ModuleNotFoundError: No module named 'onnxscript'
```

**Cause:** torchreid has many unlisted dependencies.

**Solution:**
```bash
pip install scipy opencv-python tensorboard onnx onnxscript
```

---

### Issue 3: ONNX External Weights File Missing

**Symptom:**
```
[E] Failed to open file: /path/to/osnet_x1_0.onnx.data
```

**Cause:** PyTorch exports large models as two files:
- `model.onnx` (structure only, ~900KB)
- `model.onnx.data` (weights, ~8MB)

If you only copy the `.onnx` file, TensorRT can't find the weights.

**Solution:** Convert to single file with embedded weights:
```python
import onnx

model = onnx.load('osnet_x1_0.onnx', load_external_data=True)
onnx.save(model, 'osnet_x1_0_final.onnx', save_as_external_data=False)
# Result: single 9MB file with everything included
```

---

### Issue 4: Batch Size Mismatch

**Symptom:**
```
Backend has maxBatchSize 1 whereas 16 has been requested
[E] deserialized backend context failed to match config params
```

**Cause:** TensorRT engines are optimized for specific batch sizes. The engine was built with batch=1 (default), but config requested batch=16.

**Why TensorRT engines have fixed batch sizes:**
- TensorRT pre-allocates memory for the exact batch size
- It selects CUDA kernels optimized for those dimensions
- Different batch sizes would need different optimizations

**Solution:** Change config to match engine:
```ini
batch-size=1  # Must match engine, not arbitrary
```

**Alternative:** Rebuild engine with desired batch size (only works with dynamic ONNX):
```bash
trtexec --onnx=model.onnx \
  --minShapes=input:1x3x256x128 \
  --optShapes=input:16x3x256x128 \
  --maxShapes=input:16x3x256x128 \
  --saveEngine=model_b16.engine
```

---

### Issue 5: Relative vs Absolute Paths

**Symptom:**
```
Failed to open file: models/osnet/osnet_x1_0.onnx
```

**Cause:** Relative paths resolve based on current working directory, which varies.

**Solution:** Always use absolute paths in DeepStream configs:
```ini
# Bad (relative)
onnx-file=models/osnet/osnet_x1_0.onnx

# Good (absolute)
onnx-file=/app/notebooks/reid_pipeline/models/osnet/osnet_x1_0_final.onnx
```

---

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Failed to open .onnx.data` | External weights file missing | Convert to embedded weights |
| `maxBatchSize 1 whereas 16 requested` | Batch size mismatch | Change config `batch-size=1` |
| `Failed to create NvDsInferContext` | Config error | Check paths, batch size |
| `Kernel died` in Jupyter | Segfault in C++ code | Check TensorRT build logs |
| `externally-managed-environment` | System Python protected | Use virtual environment |
| `No module named 'X'` | Missing dependency | `pip install X` |

---

## Files Reference

```
reid_pipeline/
├── README.md                         # This documentation
├── person_reid_gallery_h100.ipynb    # Main notebook
├── reid_sgie_config.txt              # SGIE configuration
├── export_osnet.py                   # Model export script
├── models/
│   └── osnet/
│       ├── osnet_x1_0_final.onnx     # Model with embedded weights
│       └── osnet_x1_0.engine         # TensorRT engine (auto-generated)
└── gallery/                          # ChromaDB storage (auto-created)
```

---

## Quick Reference Commands

```bash
# Export model (outside container)
python export_osnet.py

# Copy to container
docker cp osnet_x1_0_final.onnx container:/app/notebooks/reid_pipeline/models/osnet/

# Build TensorRT engine
docker exec container trtexec --onnx=/path/to/model.onnx --saveEngine=/path/to/model.engine --fp16

# Install ChromaDB
docker exec container pip install chromadb

# Check config
docker exec container cat /app/notebooks/reid_pipeline/reid_sgie_config.txt
```

---

## Additional Resources

- [DeepStream Python Bindings](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [torchreid Library](https://github.com/KaiyangZhou/deep-person-reid)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [ml6team DeepStream Re-ID Reference](https://github.com/ml6team/deepstream-python)

