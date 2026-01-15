#!/usr/bin/env python3
"""
LLaVA-1.5-7B with 4-bit quantization
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image


model_name = "NousResearch/Obsidian-3B-V0.5"

print("Loading LLaVA-1.5-7B with 4-bit quantization...")

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load processor
processor = AutoProcessor.from_pretrained(model_name)

# Load model with 4-bit quantization
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
)

model.eval()

print("✓ Model loaded (4-bit quantized)")

# Load image
image = Image.open("ship_2.jpg").convert("RGB")

# Prepare prompt
prompt = "USER: <image>\nDescribe the image briefly.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# Generate
print("Generating...")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# Decode
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Extract answer
if "ASSISTANT:" in result:
    answer = result.split("ASSISTANT:")[-1].strip()
else:
    answer = result

print(answer)