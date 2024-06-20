import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import random
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import is_torch2_available
from utils.gradio_utils import cal_attn_mask_xl
from utils.utils import get_comic
from utils.style_template import styles
import copy
from b2sdk.v1 import InMemoryAccountInfo, B2Api, UploadSourceLocalFile
from dotenv import load_dotenv
 
torch.cuda.empty_cache()

if is_torch2_available():
    from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils import AttnProcessor

# Global variables
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Pixar/Disney Character"
MAX_SEED = np.iinfo(np.int32).max
global models_dict
use_va = False
models_dict = {
   "Juggernaut":"RunDiffusion/Juggernaut-XL-v8",
   "RealVision":"SG161222/RealVisXL_V4.0" ,
   "SDXL":"stabilityai/stable-diffusion-xl-base-1.0" ,
   "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SpatialAttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size=None, cross_attention_dim=None, id_length=4, device="cuda", dtype=torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global total_count, attn_count, cur_step, mask1024, mask4096
        global sa32, sa64
        global write
        global height, width
        if write:
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device), hidden_states[:1], self.id_bank[cur_step][1].to(self.device), hidden_states[1:]))
        if cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length, :mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length, :mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(self.total_length, self.id_length, sa32, sa64, height, width, device=self.device, dtype=self.dtype)
        return hidden_states

    def __call1__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token, channel)
        batch_size, sequence_length, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, nums_token, channel).reshape(-1, (self.id_length + 1) * nums_token, channel)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call2__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, channel = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, sequence_length, channel).reshape(-1, (self.id_length + 1) * sequence_length, channel)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

def set_attention_processor(unet, id_length):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(attn_procs)

global attn_count, total_count, id_length, total_length, cur_step, cur_model_type
global write
global sa32, sa64
global height, width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
device = "cuda"
global attn_procs, unet
attn_procs = {}
write = False
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
global pipe
global sd_model_path
sd_model_path = models_dict["RealVision"]
pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16, use_safetensors=False)
pipe = pipe.to(device)
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
unet = pipe.unet
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None and (name.startswith("up_blocks")):
        attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
        total_count += 1
    else:
        attn_procs[name] = AttnProcessor()
print("Successfully loaded consistent self-attention")
print(f"Number of the processor: {total_count}")
unet.set_attn_processor(copy.deepcopy(attn_procs))
global mask1024, mask4096
mask1024, mask4096 = cal_attn_mask_xl(total_length, id_length, sa32, sa64, height, width, device=device, dtype=torch.float16)

guidance_scale = 4 #5
seed = 57 #57 worked best
sa32 = 0.5 #0.99
sa64 = 0.5 #0.99
id_length = 4
num_steps = 50

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

def generate_comic_from_file(file_path, output_pdf_path, output_img_dir):
    # Initialize variables to hold the general prompt and detailed prompts
    general_prompt = ""
    prompt_array = []

    # Read the file and process the content
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Flag to check if we are reading detailed prompts
        reading_detailed_prompts = False

        for line in lines:
            if line.startswith("General Prompt:"):
                general_prompt = line[len("General Prompt:"):].strip()
            elif line.startswith("Detailed Prompts:"):
                reading_detailed_prompts = True
            elif reading_detailed_prompts:
                # Extract the prompt without the index number and add to the array
                if line.strip():
                    prompt = line.split('.', 1)[1].strip()  # Remove the index number
                    prompt_array.append(prompt)

    ### Set the generated Style
    style_name = "Pixar/Disney Character"
    setup_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    prompts = [general_prompt + "," + prompt for prompt in prompt_array]
    id_prompts = prompts[:id_length]
    real_prompts = prompts[id_length:]
    torch.cuda.empty_cache()
    global write, attn_count, cur_step
    write = True
    cur_step = 0
    attn_count = 0
    negative_prompt = "naked, deformed, bad anatomy, wavy buttons, wavy screens, text, comic-book letters, low detail, less detail, bad detail, bad screen, text bubble, text box, comicbook ,comic book, dialogue, dialogues, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, extra limb, extra tail, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands"

    id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)

    id_images = pipe(id_prompts, num_inference_steps=num_steps, guidance_scale=guidance_scale, height=height, width=width, negative_prompt=negative_prompt, generator=generator).images

    write = False
    real_images = []
    for real_prompt in real_prompts:
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        real_images.append(pipe(real_prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale, height=height, width=width, negative_prompt=negative_prompt, generator=generator).images[0])

    all_images = id_images + real_images

    # Save all images to the output directory
    os.makedirs(output_img_dir, exist_ok=True)
    for idx, img in enumerate(all_images):
        img_path = os.path.join(output_img_dir, f"image_{idx}_new.png")
        img.save(img_path)
        print(f"Image saved at {img_path}")

    all_images[0].save(output_pdf_path, save_all=True, append_images=all_images[1:], quality=95)
    print(f"PDF saved at {output_pdf_path}")

# Load Backblaze credentials from files
def load_backblaze_credentials():
    with open('/home/rjayanth/StoryDiffusion/backblaze.txt', 'r') as backblaze:
        backblaze_key = backblaze.read().strip()
    with open('/home/rjayanth/StoryDiffusion/backblaze_id.txt', 'r') as backblaze:
        backblaze_id = backblaze.read().strip()
    return backblaze_id, backblaze_key

# Authorize Backblaze account
def authorize_backblaze():
    backblaze_id, backblaze_key = load_backblaze_credentials()
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", backblaze_id, backblaze_key)
    return b2_api

# Function to upload a file to Backblaze
def upload_file_to_backblaze(b2_api, bucket_name, local_file_path, remote_file_path):
    bucket = b2_api.get_bucket_by_name(bucket_name)
    source = UploadSourceLocalFile(local_file_path)
    bucket.upload(source, remote_file_path)
    print(f"Uploaded {local_file_path} to {bucket_name}/{remote_file_path}")

def main():
    base_prompt_dir = '/home/rjayanth/StoryDiffusion/generated_prompts_supercat'
    base_image_dir = '/home/rjayanth/StoryDiffusion/generated_images_supercat'
    output_dir = 'StoryDiffusion/outputs_supercat'
    os.makedirs(output_dir, exist_ok=True)

    b2_api = authorize_backblaze()
    bucket_name = 'dream-tails'

    # Process each volume
    for volume_number in range(20):  # Adjust the range as needed
        prompt_file_path = f"{base_prompt_dir}/generated_prompts_volume_{volume_number}.txt"
        output_pdf_path = os.path.join(output_dir, f"story_{volume_number}.pdf")
        output_img_dir = os.path.join(base_image_dir, f"volume_{volume_number}")

        print(f"Processing volume {volume_number}...")

        # Generate and save the comic PDF and images
        generate_comic_from_file(prompt_file_path, output_pdf_path, output_img_dir)

        # Upload PDF to Backblaze
        remote_pdf_path = f"supercat_volume_{volume_number}/story_{volume_number}.pdf"
        upload_file_to_backblaze(b2_api, bucket_name, output_pdf_path, remote_pdf_path)

        # Upload images to Backblaze
        for img_file in os.listdir(output_img_dir):
            local_img_path = os.path.join(output_img_dir, img_file)
            remote_img_path = f"supercat_volume_{volume_number}/{img_file}"
            upload_file_to_backblaze(b2_api, bucket_name, local_img_path, remote_img_path)

if __name__ == "__main__":
    main()
