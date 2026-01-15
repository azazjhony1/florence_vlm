#!/usr/bin/env python3
"""
Baseline: Original Florence-2 model without optimizations
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def get_model_and_processor(device="cpu"):
    """
    Load the baseline Florence-2 model
    
    Returns:
        tuple: (model, processor, description)
    """
    print("Loading BASELINE model...")
    
    model_name = "microsoft/Florence-2-base"
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    
    model.eval()
    
    description = "Baseline Florence-2-base model (FP32, full resolution)"
    
    return model, processor, description


if __name__ == "__main__":
    # Test loading
    model, processor, desc = get_model_and_processor()
    print(f"✓ Loaded: {desc}")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")