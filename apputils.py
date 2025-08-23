import numpy as np
import cv2
from PIL import Image
from utils_mask import get_mask_location
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import apply_net
import torch
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
from typing import List

tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


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


def start_tryon(
    dict,
    garm_img,
    garment_des,
    is_checked,
    is_checked_crop,
    denoise_steps,
    seed,
    selected_body_part,
    openpose_model,
    pipe,
    device,
    parsing_model,
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

    # if lower body is selected, read cloth_b.png as garm_img
    if selected_body_part == "lower_body":
        # process with CatVTON here
        # read cat_result.png for person image
        human_img = Image.open("cat_result.png").convert("RGB")

        human_mask = None
        try:
            human_mask = Image.open("mask_b.png").convert("RGB")
        except:
            print("mask_b.png not found")
            human_mask = None

        garm_img = Image.open("cloth_b.png").convert("RGB")

    elif selected_body_part == "dresses":
        # process with CatVTON here
        # read cat_result.png as final result and return
        human_img = Image.open("cat_result.png").convert("RGB")
        garm_img = Image.open("cloth_u.png").convert("RGB")

    else:
        garm_img = Image.open("cloth_u.png").convert("RGB")

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
