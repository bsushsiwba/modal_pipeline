import numpy as np
from PIL import Image
import cv2
import numpy as np
import os


def apply_mask_and_save(mask, ref_image, output_name):
    # Convert NumPy array to PIL image first
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    # Load and binarize the mask (white = 255, black = 0)
    mask = mask.convert("L")
    binary_mask = np.array(mask) > 128  # Boolean mask

    # Convert ref image to array
    ref_np = np.array(ref_image)

    # Create white background
    output_np = np.ones_like(ref_np) * 255  # White background

    # Apply mask
    output_np[binary_mask] = ref_np[binary_mask]

    # Convert to image and save
    output_img = Image.fromarray(output_np)
    output_img.save(output_name)


def samutils_segment(base_model):
    try:
        # delete all png files in directory except garment.png and human.png
        for file in os.listdir("."):
            if file not in [
                "garment.png",
                "human.png",
            ] and file.endswith(".png"):
                os.remove(file)

        # saving the source image's bottom mask
        img = Image.open("human.png")
        results = base_model.predict(img)
        mask_b = None
        # asigning mask
        try:
            mask_b = Image.fromarray(results.mask[2])
        except:
            pass

        if mask_b is not None:
            mask_b.save("mask_b.png")

        # read the image
        img = Image.open("garment.png").convert("RGB")

        # Predict results on image
        results = base_model.predict(img)

        mask_u = None
        mask_b = None

        # asigning masks
        try:
            mask_u = Image.fromarray(results.mask[1])
            mask_b = Image.fromarray(results.mask[2])
        except:
            print("error in assigning masks")

        if mask_u is None:
            # save image as cloth_u and cloth_b in case nothing is found
            img.save("cloth_u.png")
            img.save("cloth_b.png")
            print("No cloth found")

        elif mask_b is None or np.array_equal(mask_u, mask_b):
            # Apply and save each masked part
            apply_mask_and_save(mask_u, img, "cloth_u.png")
            apply_mask_and_save(mask_u, img, "cloth_b.png")
            print("Only one cloth found")
        else:
            # invert mask_b
            inv_mask_b = mask_b.point(lambda x: 255 - x)
            # save mask_b
            inv_mask_b.save("cloth_u.png")

            # Load pil image in grayscale
            image = cv2.imread("cloth_u.png", cv2.IMREAD_GRAYSCALE)

            # Find the first row with any black pixel (pixel value < 255)
            start_row = None
            for row in range(image.shape[0]):
                if np.any(image[row] < 255):
                    start_row = row
                    break

            # If we found the row, set everything from that row to the bottom as black
            if start_row is not None:
                image[start_row:] = 0

            mask_u.save("cloth_u.png")

            # take or with mask_u
            top = cv2.bitwise_or(image, cv2.imread("cloth_u.png", cv2.IMREAD_GRAYSCALE))

            # invert image and save
            bottom = cv2.bitwise_not(image)

            apply_mask_and_save(top, img, "cloth_u.png")
            apply_mask_and_save(bottom, img, "cloth_b.png")
    except:
        print("Error in SAM segmentation")
    return True
