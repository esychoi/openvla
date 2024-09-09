from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import requests
import torch


# Load Processor & VLA
MODEL_PATH = "/data/models/openvla-7b"
DEVICE = "cuda:2"
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
).to(DEVICE)

# Grab image input & format prompt
image_url = "/data/sample/Real/R_D_1_3/RGB1/0.png"
image = Image.open(image_url)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:2", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(action)