# Load model directly
from transformers import AutoModelForCausalLM

import requests
import torch
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cpu"
torch_dtype = torch.float32


model = AutoModelForCausalLM.from_pretrained(
    "Efficient-Large-Model/VILA-2.7b",
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    attn_implementation="eager"
).to(device)