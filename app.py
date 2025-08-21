import os
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model1.cloth_masker import AutoMasker
from model1.pipeline import CatVTONPipeline
from utils import init_weight_dtype, process_single_request

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


if __name__ == "__main__":
    temp = process_single_request(
        automasker,
        mask_processor,
        pipeline,
        Image.open("human.png"),
        Image.open("garment.png"),
        "overall",
    )

    if temp:
        temp.save("cat_result.png")
        print("Image processed and saved as output_image.png")
