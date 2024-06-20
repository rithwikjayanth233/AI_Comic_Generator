import os
import json
import random
import torch
from torch import Generator
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from utils.style_template import styles
from datetime import datetime

# Define the path and setup directories
output_dir = "/home/rjayanth/StoryDiffusion/outputs_try"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the diffusion model and other parameters
device = "cuda"
model_id = "SG161222/RealVisXL_V4.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=False).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Define the range for seeds and guidance scales
seed_range = range(1, 10001)
guidance_scale_range = [i * 0.5 for i in range(2, 15)]  # From 1 to 7 in increments of 0.5

# Initialize a set to keep track of used combinations to ensure no repetition
used_combinations = set()

# Load prompts from a file
file_path = "/home/rjayanth/StoryDiffusion/generated_prompts_unicorninja/generated_prompts_volume_11.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()
general_prompt = ""
prompt_array = []
reading_detailed_prompts = False
for line in lines:
    if line.startswith("General Prompt:"):
        general_prompt = line[len("General Prompt:"):].strip()
    elif line.startswith("Detailed Prompts:"):
        reading_detailed_prompts = True
    elif reading_detailed_prompts:
        if line.strip():
            prompt = line.split('.', 1)[1].strip()  # Remove the index number
            prompt_array.append(prompt)

# Loop to generate multiple stories
num_stories = 50  # Define how many stories you want to generate
for _ in range(num_stories):
    while True:
        seed = random.choice(seed_range)
        guidance_scale = random.choice(guidance_scale_range)
        combination = (seed, guidance_scale)
        if combination not in used_combinations:
            used_combinations.add(combination)
            break
    
    # Set seed for reproducibility
    generator = Generator(device=device).manual_seed(seed)

    # Apply the style (if needed)
    style_name = "Pixar/Disney Character"  # As an example
    styled_prompts = [styles.get(style_name, {}).get('prompt', '').format(prompt=general_prompt)] + \
                     [styles.get(style_name, {}).get('prompt', '').format(prompt=p) for p in prompt_array]

    # Generate images and create PDF
    images = pipe(styled_prompts, num_inference_steps=50, guidance_scale=guidance_scale, generator=generator).images
    pdf_path = os.path.join(output_dir, f"story_{seed}_{guidance_scale}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    images[0].save(pdf_path, save_all=True, append_images=images[1:], quality=95)

    # Save parameters to a JSON file
    params = {
        "seed": seed,
        "guidance_scale": guidance_scale,
        "prompts": styled_prompts
    }
    json_path = os.path.join(output_dir, f"params_{seed}_{guidance_scale}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, 'w') as fp:
        json.dump(params, fp)

    print(f"Processed story with seed {seed} and guidance scale {guidance_scale}. PDF saved to {pdf_path}. Parameters saved to {json_path}.")
