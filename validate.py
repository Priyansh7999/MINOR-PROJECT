from pathlib import Path
import pickle
import face_recognition
from PIL import Image, ImageDraw
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    tolerance: float = 0.4,
) -> dict:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    results = {}
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings, tolerance)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)
        results[name] = results.get(name, 0) + 1

    del draw
    pillow_image.show()

    return results

def _recognize_face(unknown_encoding, loaded_encodings, tolerance):
    distances = face_recognition.face_distance(loaded_encodings["encodings"], unknown_encoding)
    min_distance = min(distances)
    if min_distance <= tolerance:
        index = distances.argmin()
        return loaded_encodings["names"][index]
    return None

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )

def validate(image_location: str, model: str = "hog", tolerance: float = 0.4):
    results = recognize_faces(
        image_location=image_location,
        model=model,
        tolerance=tolerance
    )
    logging.info(f"Validation results for {Path(image_location).name}:")
    if not results:
        logging.info("  No faces found")
    else:
        for name, count in results.items():
            logging.info(f"  {name}: count = {count}")

# call vadidate("img.jpg")
# validate()
