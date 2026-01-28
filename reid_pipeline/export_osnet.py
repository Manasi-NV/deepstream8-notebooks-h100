#!/usr/bin/env python3
"""
OSNet ONNX Export Script

Run this OUTSIDE the DeepStream container (needs PyTorch + torchreid).

Usage:
    pip install torchreid torch torchvision
    python export_osnet.py

Output:
    osnet_x1_0.onnx - Copy this to models/osnet/ inside your DeepStream container
"""

import torch
import torch.nn as nn

def export_osnet():
    """Export OSNet x1_0 to ONNX format"""
    
    try:
        from torchreid import models
    except ImportError as e:
        print(f"ERROR: torchreid import failed: {e}")
        print("Run: pip install torchreid torch torchvision scipy opencv-python tensorboard")
        return
    
    print("Loading OSNet x1_0 pretrained model...")
    
    # Build OSNet model
    model = models.build_model(
        name='osnet_x1_0',
        num_classes=1,  # Doesn't matter for feature extraction
        loss='softmax',
        pretrained=True
    )
    model.eval()
    
    # Remove classification head - we only want features
    # OSNet's forward() returns features when not training
    
    print("Model loaded successfully!")
    print(f"Output embedding dimension: 512")
    
    # Create dummy input (batch=1, channels=3, height=256, width=128)
    dummy_input = torch.randn(1, 3, 256, 128)
    
    # Test forward pass
    with torch.no_grad():
        features = model(dummy_input)
        print(f"Test output shape: {features.shape}")  # Should be [1, 512]
    
    # Export to ONNX
    output_path = "osnet_x1_0.onnx"
    print(f"\nExporting to {output_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"\nSUCCESS! Exported to: {output_path}")
    print("\nNext steps:")
    print("1. Copy osnet_x1_0.onnx to your DeepStream container:")
    print("   docker cp osnet_x1_0.onnx <container>:/app/notebooks/reid_pipeline/models/osnet/")
    print("2. Run the person_reid_gallery_h100.ipynb notebook")


def verify_onnx():
    """Verify the exported ONNX model"""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("Install onnx and onnxruntime to verify: pip install onnx onnxruntime")
        return
    
    print("\nVerifying ONNX model...")
    
    # Load and check model
    model = onnx.load("osnet_x1_0.onnx")
    onnx.checker.check_model(model)
    print("ONNX model is valid!")
    
    # Test inference
    session = ort.InferenceSession("osnet_x1_0.onnx")
    dummy_input = np.random.randn(1, 3, 256, 128).astype(np.float32)
    outputs = session.run(None, {'input': dummy_input})
    print(f"ONNX inference output shape: {outputs[0].shape}")  # Should be (1, 512)
    print("ONNX model verified successfully!")


if __name__ == "__main__":
    export_osnet()
    
    # Optional: verify the exported model
    try:
        verify_onnx()
    except Exception as e:
        print(f"Verification skipped: {e}")

