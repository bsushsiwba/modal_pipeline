import gradio as gr
import io
from PIL import Image
import torch
import os
import time
import subprocess
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import uvicorn
from pyngrok import ngrok, conf
from enum import Enum

# import your existing pipeline setup here
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from apputils import start_tryon

conf.get_default().auth_token = "2I5wsdcXFHV3hMVNmc3Ki8ifZPi_4GJEBnPpHpVQkdSPfFCuz"

# ---------------------------
# INIT MODELS ONCE
# ---------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
base_path = "yisol/IDM-VTON"

# background processes
subprocess.Popen(["python", "sam_invoke.py"])
subprocess.Popen(["python", "cat_invoke.py"])

# main components
unet = UNet2DConditionModel.from_pretrained(
    base_path, subfolder="unet", torch_dtype=torch.float16
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    base_path, subfolder="tokenizer", use_fast=False
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path, subfolder="tokenizer_2", use_fast=False
)

noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
text_encoder_one = CLIPTextModel.from_pretrained(
    base_path, subfolder="text_encoder", torch_dtype=torch.float16
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path, subfolder="text_encoder_2", torch_dtype=torch.float16
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path, subfolder="image_encoder", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(
    base_path, subfolder="vae", torch_dtype=torch.float16
)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path, subfolder="unet_encoder", torch_dtype=torch.float16
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

for m in [UNet_Encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two]:
    m.requires_grad_(False)

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


# ---------------------------
# TRYON FUNCTION
# ---------------------------
def run_tryon(human_image: Image.Image, garment_image: Image.Image, garment_type: str):
    # delete old marker files
    for f in [
        "process_sam.txt",
        "cat_full.txt",
        "cat_lower.txt",
        "sam_complete.txt",
        "cat_complete.txt",
    ]:
        if os.path.exists(f):
            os.remove(f)

    # validate garment type
    if garment_type not in ["upper", "lower", "full"]:
        raise ValueError(
            "Invalid garment_type. Must be one of: 'upper', 'lower', 'full'."
        )
    elif garment_type == "upper":
        garment_type = "upper_body"
    elif garment_type == "lower":
        garment_type = "lower_body"
    elif garment_type == "full":
        garment_type = "dresses"

    # save input images
    human_image.save("human.png")
    garment_image.save("garment.png")

    # trigger SAM
    with open("process_sam.txt", "w") as f:
        f.write("process sam")

    # trigger CAT
    with open(f"cat_{'full' if garment_type == 'dresses' else 'lower'}.txt", "w") as f:
        f.write("process cat")

    # wait for signals
    while not (
        os.path.exists("sam_complete.txt") and os.path.exists("cat_complete.txt")
    ):
        time.sleep(0.1)

    # reload images
    human_img = Image.open("human.png").convert("RGB")
    garm_img = Image.open("garment.png").convert("RGB")

    dummy_dict = {"background": human_img, "layers": [human_img]}

    print("Calling IDM Tryon_Pipeline...")
    result, _ = start_tryon(
        dummy_dict,
        garm_img,
        "",  # description
        True,  # is_checked
        True,  # is_checked_crop
        30,  # denoise_steps
        42,  # seed
        garment_type,
        openpose_model,
        pipe,
        device,
        parsing_model,
    )

    return result


app = FastAPI()


class DressingType(str, Enum):
    upper = "upper"
    lower = "lower"
    full = "full"


@app.post("/tryon")
async def tryon(
    human_image: UploadFile,
    garment_image: UploadFile,
    dressing_type: DressingType = Form(...),
):
    # Read images into bytes
    human_bytes = await human_image.read()
    print("[tryon] Read human image bytes.")
    garment_bytes = await garment_image.read()
    print("[tryon] Read garment image bytes.")

    result = run_tryon(
        Image.open(io.BytesIO(human_bytes)).convert("RGB"),
        Image.open(io.BytesIO(garment_bytes)).convert("RGB"),
        dressing_type.value,
    )

    return StreamingResponse(io.BytesIO(result), media_type="image/png")


if __name__ == "__main__":
    # Open an ngrok tunnel to the FastAPI app
    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8000)
