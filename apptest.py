from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import time


device = "cuda:0" if torch.cuda.is_available() else "cpu"


import cv2
import numpy as np
from PIL import Image

from apputils import resize_and_center, pil_to_binary_mask, start_tryon

import os
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model1.cloth_masker import AutoMasker
from model1.pipeline import CatVTONPipeline
from utils import init_weight_dtype, process_single_request


base_path = "yisol/IDM-VTON"
example_path = os.path.join(os.path.dirname(__file__), "example")

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    base_path,
    subfolder="vae",
    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder


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


# Main block: process upper_body using human.png and garment.png, save result, and exit
if __name__ == "__main__":
    try:
        # Read images
        human_img = Image.open("human.png").convert("RGB")
        garm_img = Image.open("garment.png").convert("RGB")

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

        # Create dummy dict for compatibility
        dict = {"background": human_img, "layers": [human_img]}

        # Process images using existing pipeline for upper_body
        result, _ = start_tryon(
            dict,
            garm_img,
            "",  # Generic description
            True,  # is_checked
            True,  # is_checked_crop
            30,  # denoise_steps
            42,  # seed
            "upper_body",
            openpose_model,
            pipe,
            device,
            parsing_model,
        )

        # Save result
        result.save("result.png")
        print("Processing complete. Saved to ../result.png")
    except Exception as e:
        print(f"Error processing files: {e}")
