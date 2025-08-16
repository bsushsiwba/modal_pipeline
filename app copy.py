import sys

sys.path.append("./")
from PIL import Image
import gradio as gr
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
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
from torchvision.transforms.functional import to_pil_image
import time
import threading


device = "cuda:0" if torch.cuda.is_available() else "cpu"


import cv2
import numpy as np
from PIL import Image


def resize_and_center(image, target_width, target_height):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    original_height, original_width = img.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    resized_img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )

    padded_img = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2

    padded_img[top : top + new_height, left : left + new_width] = resized_img

    return Image.fromarray(padded_img)


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in binary_mask.shape[1]:
            if binary_mask[i, j] == True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


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
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

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


def start_tryon(
    dict,
    garm_img,
    garment_des,
    is_checked,
    is_checked_crop,
    denoise_steps,
    seed,
    selected_body_part,
):

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = resize_and_center(dict["background"], 768, 1024)
    human_img_orig = human_img_orig.convert("RGB")

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    # save garment and human image
    garm_img.save("../vesti-backend-sam/garment.png")
    human_img.save("../vesti-backend-sam/human.png")

    if os.path.exists("../vesti-backend-sam/complete.txt"):
        os.remove("../vesti-backend-sam/complete.txt")

    with open("../vesti-backend-sam/process.txt", "w") as f:
        f.write("1")

    # wait while complete.txt is not created
    while not os.path.exists("../vesti-backend-sam/complete.txt"):
        time.sleep(0.1)

    # if lower body is selected, read cloth_b.png as garm_img
    if selected_body_part == "lower_body":
        # process with CatVTON here
        # delete all png files in CatVTON directory
        for file in os.listdir("../CatVTON"):
            if file.endswith(".png"):
                os.remove(os.path.join("../CatVTON", file))

        # save the person and cloth images
        human_img.save("../CatVTON/person_image.png")
        garm_img.save("../CatVTON/cloth_image.png")

        # save lower.txt to initiate processing
        with open("../CatVTON/lower.txt", "w") as f:
            f.write("process")

        # wait while complete.txt is not created
        while not os.path.exists("../CatVTON/complete.txt"):
            time.sleep(0.1)

        # delete complete.txt
        if os.path.exists("../CatVTON/complete.txt"):
            os.remove("../CatVTON/complete.txt")

        # read cat_result.png for person image
        human_img = Image.open("../CatVTON/cat_result.png").convert("RGB")

        human_mask = None
        try:
            human_mask = Image.open("../vesti-backend-sam/mask_b.png").convert("RGB")
        except:
            print("mask_b.png not found")
            human_mask = None

        garm_img = Image.open("../vesti-backend-sam/cloth_b.png").convert("RGB")

    elif selected_body_part == "dresses":
        # process with CatVTON here
        # delete all png files in CatVTON directory
        for file in os.listdir("../CatVTON"):
            if file.endswith(".png"):
                os.remove(os.path.join("../CatVTON", file))

        # save the person and cloth images
        human_img.save("../CatVTON/person_image.png")
        garm_img.save("../CatVTON/cloth_image.png")

        # save full.txt to initiate processing
        with open("../CatVTON/full.txt", "w") as f:
            f.write("process")

        # wait while complete.txt is not created
        while not os.path.exists("../CatVTON/complete.txt"):
            time.sleep(0.1)

        # delete complete.txt
        if os.path.exists("../CatVTON/complete.txt"):
            os.remove("../CatVTON/complete.txt")

        # read cat_result.png as final result and return
        human_img = Image.open("../CatVTON/cat_result.png").convert("RGB")
        garm_img = Image.open("../vesti-backend-sam/cloth_u.png").convert("RGB")

    else:
        garm_img = Image.open("../vesti-backend-sam/cloth_u.png").convert("RGB")

    garm_img = garm_img.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location(
            "hd", selected_body_part, model_parse, keypoints
        )

        # if selected lower body then or with human_mask
        if selected_body_part == "lower_body" and human_mask is not None:
            # Convert human_mask to grayscale and ensure 2D array
            human_mask = human_mask.convert("L")
            human_mask = human_mask.resize((mask.size[0], mask.size[1]))
            mask = Image.fromarray(
                np.clip(
                    np.array(mask, dtype=np.uint8)
                    | np.array(human_mask, dtype=np.uint8),
                    0,
                    255,
                ).astype(np.uint8)
            )

            # go column wise, note first and last white pixel. Fill all pixels in between with white
            mask = np.array(mask)
            for j in range(mask.shape[1]):  # loop through columns
                first_white = -1
                last_white = -1
                for i in range(mask.shape[0]):  # loop through rows
                    if mask[i, j] == 255:
                        if first_white == -1:
                            first_white = i
                        last_white = i
                if first_white != -1 and last_white != -1:
                    mask[first_white:last_white, j] = 255
            mask = Image.fromarray(mask)
            mask = mask.convert("L")
        mask = mask.resize((768, 1024))

    else:
        mask = pil_to_binary_mask(dict["layers"][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)

    # get last white pixel in each column, if it is less than 50px from bottom, set all from there to bottom to white
    mask = np.array(mask)
    for j in range(mask.shape[1]):  # loop through columns
        last_white = -1
        for i in range(mask.shape[0] - 1, -1, -1):  # loop through rows
            if mask[i, j] == 255:
                last_white = i
                break
        if last_white != -1 and last_white > mask.shape[0] - 50:
            mask[last_white:, j] = 255
    mask = Image.fromarray(mask)
    mask = mask.convert("L")

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "./ckpt/densepose/model_final_162be9.pkl",
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            "cuda",
        )
    )
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = (
                    "monochrome, lowres, bad anatomy, worst quality, low quality"
                )
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = "a photo of " + garment_des
                    negative_prompt = (
                        "monochrome, lowres, bad anatomy, worst quality, low quality"
                    )
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = (
                        tensor_transfrom(pose_img)
                        .unsqueeze(0)
                        .to(device, torch.float16)
                    )
                    garm_tensor = (
                        tensor_transfrom(garm_img)
                        .unsqueeze(0)
                        .to(device, torch.float16)
                    )
                    generator = (
                        torch.Generator(device).manual_seed(seed)
                        if seed is not None
                        else None
                    )
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(
                            device, torch.float16
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(
                            device, torch.float16
                        ),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                            device, torch.float16
                        ),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor.to(device, torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768, 1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray
    # return images[0], mask_gray


garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict = {}
    ex_dict["background"] = ex_human
    ex_dict["layers"] = None
    ex_dict["composite"] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## Vesti Demo")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(
                sources="upload",
                type="pil",
                label="Human. Mask with pen or use auto-masking",
                interactive=True,
            )
            is_checked = gr.State(value=True)  # Hidden state variable
            is_checked_crop = gr.State(value=True)  # Hidden state variable
            # Hidden advanced settings
            denoise_steps = gr.State(value=30)
            seed = gr.State(value=42)
            with gr.Row():
                body_part = gr.Dropdown(
                    choices=["upper_body", "lower_body", "dresses"],
                    value="upper_body",
                    label="Select Body Part",
                    info="Choose the type of clothing",
                )

            example = gr.Examples(
                inputs=imgs, examples_per_page=10, examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources="upload", type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts",
                        show_label=False,
                        elem_id="prompt",
                    )
            example = gr.Examples(
                inputs=garm_img, examples_per_page=8, examples=garm_list_path
            )
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(
                label="Masked image output",
                elem_id="masked-img",
                show_share_button=False,
            )
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(
                label="Output", elem_id="output-img", show_share_button=False
            )

    with gr.Column():
        try_button = gr.Button(value="Try-on")

    try_button.click(
        fn=start_tryon,
        inputs=[
            imgs,
            garm_img,
            prompt,
            is_checked,
            is_checked_crop,
            denoise_steps,
            seed,
            body_part,
        ],
        outputs=[image_out, masked_img],
        api_name="tryon",
    )


def process_from_file():
    valid_types = ["upper_body", "lower_body", "dresses"]
    while True:
        if os.path.exists("../process.txt"):
            try:
                # Read dress type from process.txt
                with open("../process.txt", "r") as f:
                    selected_body_part = f.read().strip()

                # Default to upper_body if empty or invalid
                if not selected_body_part:
                    selected_body_part = "upper_body"
                elif selected_body_part not in valid_types:
                    raise ValueError(f"Invalid clothing type. Must be one of: {valid_types}")

                # Remove process.txt
                os.remove("../process.txt")

                # Read images
                human_img = Image.open("../human.png").convert("RGB")
                garm_img = Image.open("../garment.png").convert("RGB")

                # Create dummy dict for compatibility
                dict = {"background": human_img, "layers": [human_img]}

                # Process images using existing pipeline
                result, _ = start_tryon(
                    dict,
                    garm_img,
                    "",  # Generic description
                    True,  # is_checked
                    True,  # is_checked_crop
                    30,  # denoise_steps
                    42,  # seed
                    selected_body_part,
                )

                # Save result
                result.save("../result.png")

                # Create complete.txt
                with open("../complete.txt", "w") as f:
                    f.write("done")

            except Exception as e:
                print(f"Error processing files: {e}")
                # Create complete.txt with error
                with open("../complete.txt", "w") as f:
                    f.write(f"error: {str(e)}")

        time.sleep(0.1)  # Check every second


# Start file processing thread
process_thread = threading.Thread(target=process_from_file, daemon=True)
process_thread.start()

image_blocks.launch(share=True)
