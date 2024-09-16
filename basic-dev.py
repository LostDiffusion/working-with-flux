# From https://huggingface.co/black-forest-labs/FLUX.1-dev
# Changes by Rune Bloodstone\
# This does not run on less than 8 GB of RAM. It produces an error.
# The Flux-1.dev model is a Gated model on HuggingFace.

import torch
from diffusers import FluxPipeline
from datetime import datetime

image_name = "flux-dev.png"
start_time = datetime.now()

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() # Save some VRAM.

prompt = "A black cat holding a BlackBerry Bold phone in its cute paws standing in front of a building with a corporate BlackBerry sign on it."

image = pipe(prompt, height=1024, width=1024, 
            guidance_scale=3.5, num_inference_steps=50, max_sequence_length=512, 
            generator=torch.Generator("cpu").manual_seed(0)).images[0]

stop_time = datetime.now()

time_diff = stop_time - start_time
print(f"[+] Image generated in {time_diff} ms.")

print("[+] Saving image...", end=" ")
image.save(image_name)

print("done.")
