import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cpu"
torch_dtype = torch.float32

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    attn_implementation="eager"
).to(device)

# Load processor
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", 
    trust_remote_code=True
)

# Define prompt
prompt = "<OD>"

# Load image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

# Process inputs - IMPORTANT: note the correct syntax
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

# Generate - IMPORTANT: use the exact parameters from the tutorial
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
    use_cache=False
)

# Decode
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# Post-process
parsed_answer = processor.post_process_generation(
    generated_text, 
    task="<OD>", 
    image_size=(image.width, image.height)
)

print(parsed_answer)