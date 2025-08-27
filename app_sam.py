from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import uvicorn
from pyngrok import ngrok
import subprocess
import os
import time
import io

subprocess.Popen(["python", "sam_invoke.py"])

app = FastAPI()


@app.post("/tryon")
async def tryon(
    human_image: UploadFile,
    garment_image: UploadFile,
):
    # Read images into bytes
    human_bytes = await human_image.read()
    print("[tryon] Read human image bytes.")
    garment_bytes = await garment_image.read()
    print("[tryon] Read garment image bytes.")

    # save images to disk
    with open("human.png", "wb") as f:
        f.write(human_bytes)
    print("[tryon] Saved human image to disk.")
    with open("garment.png", "wb") as f:
        f.write(garment_bytes)
    print("[tryon] Saved garment image to disk.")

    # trigger SAM
    with open("process_sam.txt", "w") as f:
        f.write("process sam")

    # wait for signals
    while not (os.path.exists("sam_complete.txt")):
        time.sleep(0.1)

    # read cloth_u.png
    with open("cloth_u.png", "rb") as f:
        cloth_bytes = f.read()

    return StreamingResponse(io.BytesIO(cloth_bytes), media_type="image/png")


if __name__ == "__main__":
    # Open an ngrok tunnel to the FastAPI app
    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8000)
