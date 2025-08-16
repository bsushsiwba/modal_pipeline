from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from PIL import Image
import io
import sys

sys.path.append("./gradio_demo")
from gradio_demo.app import start_tryon
import torch

app = FastAPI()


@app.post("/tryon")
async def try_on(
    human: UploadFile = File(...),
    garment: UploadFile = File(...),
    body_part: str = Form(...),
    garment_description: str = Form(...),
):
    # Read and convert uploaded files to PIL Images
    human_image = Image.open(io.BytesIO(await human.read()))
    garment_image = Image.open(io.BytesIO(await garment.read()))

    # Prepare the dictionary input expected by start_tryon
    human_dict = {
        "background": human_image,
        "layers": [Image.new("RGB", human_image.size, "white")],  # Create a white mask
        "composite": None,
    }

    # Default parameters for try-on
    is_checked = True
    is_checked_crop = True
    denoise_steps = 30
    seed = 42

    # Process the images
    result_image, mask_image = start_tryon(
        human_dict,
        garment_image,
        garment_description,
        is_checked,
        is_checked_crop,
        denoise_steps,
        seed,
        body_part,
    )

    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
