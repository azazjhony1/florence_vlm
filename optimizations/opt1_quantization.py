#!/usr/bin/env python3
"""
Optimization 1: Dynamic INT8 Quantization
Expected speedup: 2-3x on CPU
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def get_model_and_processor(device="cpu"):
    """
    Load dynamically quantized INT8 model
    
    Returns:
        tuple: (model, processor, description)
    """
    print("Loading INT8 QUANTIZED model...")
    
    model_name = "microsoft/Florence-2-base"
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    # Load original model
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    # Apply dynamic quantization (fast, happens in-place)
    print("  Applying INT8 quantization...")
    
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize all linear layers
        dtype=torch.qint8
    )
    
    quantized_model = quantized_model.to(device)
    quantized_model.eval()
    
    print("  ✓ Quantization complete")
    
    description = "INT8 Dynamic Quantization (~2-3x faster on CPU)"
    
    return quantized_model, processor, description


if __name__ == "__main__":
    # Test loading and basic inference
    print("Testing INT8 quantization...")
    
    model, processor, desc = get_model_and_processor()
    print(f"\n✓ Loaded: {desc}")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    # Quick inference test
    try:
        from PIL import Image
        import requests
        from io import BytesIO
        
        print("\nRunning quick inference test...")
        
        # Download sample image
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        # Process
        inputs = processor(text="<CAPTION>", images=image, return_tensors="pt")
        
        # Generate
        import time
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, use_cache=False)
        inference_time = time.perf_counter() - start
        
        # Decode
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Result: {result}")
        print("\n✓ Quantized model works correctly!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()