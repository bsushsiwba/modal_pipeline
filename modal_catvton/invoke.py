from app import process_single_request

if __name__ == "__main__":
    process_single_request(
        person_image="path/to/person_image.jpg",
        garment_image="path/to/garment_image.jpg",
        garment_type="overall",
    )
