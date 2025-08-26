import os
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import time

from model1.cloth_masker import AutoMasker
from model1.pipeline import CatVTONPipeline
from catutils import init_weight_dtype, process_single_request

repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("bf16"),
    use_tf32=True,
    device="cuda",
    skip_safety_check=True,
)
# AutoMasker
mask_processor = VaeImageProcessor(
    vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device="cuda",
)

# open images
human_img = Image.open("human.png")
garm_img = Image.open("garment.png")

# wait for cat_full or cat_lower to be created
while not (os.path.exists("cat_full.txt") or os.path.exists("cat_lower.txt")):
    time.sleep(0.1)

garm_type = "full" if os.path.exists("cat_full.txt") else "lower"

# delete cat_full.txt and cat_lower.txt if they exist
if os.path.exists("cat_full.txt"):
    os.remove("cat_full.txt")
if os.path.exists("cat_lower.txt"):
    os.remove("cat_lower.txt")

temp = process_single_request(
    automasker,
    mask_processor,
    pipeline,
    human_img,
    garm_img,
    garm_type,
)

if temp:
    temp.save("cat_result.png")
    print("Image processed and saved as output_image.png")

# create cat_complete.txt to signal completion
with open("cat_complete.txt", "w") as f:
    f.write("done")
