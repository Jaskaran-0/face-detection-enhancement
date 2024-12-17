import os
import cv2
import shutil
import re
import gc
import torch
from deepface import DeepFace
import numpy as np

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants for maximum window size
MAX_WINDOW_WIDTH = 1200
MAX_WINDOW_HEIGHT = 800
WINDOW_POSITION_X = 100
WINDOW_POSITION_Y = 100

# Constants for website image dimensions
WEBSITE_WIDTH = 1800
WEBSITE_HEIGHT = 800
SQL_FILE = "face_labels1.sql"
TABLE_NAME = "face_bounding_boxes"

# Constant for preview box size
PREVIEW_BOX_SIZE = 500

# Define directories
base_dir = "photos1"
faces_dir = "faces"
labeled_faces_dir = "labeled_faces1"
finished_photos_dir = "finished_photos1"
annotated_photos_dir = "annotated_photos1"

# Create directories if they don't exist
os.makedirs(faces_dir, exist_ok=True)
os.makedirs(labeled_faces_dir, exist_ok=True)
os.makedirs(finished_photos_dir, exist_ok=True)
os.makedirs(annotated_photos_dir, exist_ok=True)


# Function to resize the image to fit a maximum target size (20MB)
def resize_image_opencv(image, target_size=20 * 1024 * 1024):
    """Resize the image using OpenCV to fit within the target file size."""
    encode_param = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    quality = 95
    current_size = len(cv2.imencode('.tif', image, encode_param)[1].tobytes())

    while current_size > target_size:
        # Reduce the quality of the image to reduce file size
        quality -= 5
        if quality < 20:  # Avoid reducing quality below acceptable levels
            break

        # Resize by reducing the width and height
        height, width = image.shape[:2]
        scaling_factor = 0.9  # Reduce by 10% each iteration
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Update the current size after resizing
        current_size = len(
            cv2.imencode('.tif', image, encode_param)[1].tobytes())

    return image


# Function to scale coordinates after resizing
def scale_coordinates(orig_dim, new_dim, coord):
    """Scale coordinates to match resized dimensions."""
    scale_x = new_dim[0] / orig_dim[0]
    scale_y = new_dim[1] / orig_dim[1]
    return int(coord[0] * scale_x), int(coord[1] * scale_y), int(
        coord[2] * scale_x), int(coord[3] * scale_y)


# Function to generate SQL query for storing face bounding boxes
def generate_sql(face_id, photo_name, x, y, w, h, label):
    """Generate SQL for face bounding box."""
    return f"INSERT INTO {TABLE_NAME} (face_id, photo_name, x, y, width, height, label) VALUES ('{face_id}', '{photo_name}', {x}, {y}, {w}, {h}, '{label}');\n"


# Function to detect and crop faces from an image
def detect_and_crop_faces(photo_path, faces_dir):
    """Detect faces, crop them, and label."""
    print(f"Processing {photo_path}")
    img = cv2.imread(photo_path)

    # UPDATED: Resize the image to make it more manageable
    img = resize_image_opencv(img)  # Resize the image to fit under 20MB

    orig_height, orig_width = img.shape[:2]

    # UPDATED: Use CUDA if available
    try:
        detected_faces = DeepFace.extract_faces(img, detector_backend='mtcnn',
                                                align=True,
                                                enforce_detection=False)
        print(
            f"Total faces detected in {os.path.basename(photo_path)}: {len(detected_faces)}")
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        return

    sql_statements = []
    face_id = 1
    unknown_counter = get_next_unknown_number(labeled_faces_dir)

    # Order faces from top to bottom, left to right
    detected_faces.sort(
        key=lambda face: (face["facial_area"]["y"], face["facial_area"]["x"]))

    # Scale the entire image to website size
    scaled_img = cv2.resize(img, (WEBSITE_WIDTH, WEBSITE_HEIGHT),
                            interpolation=cv2.INTER_AREA)

    for face in detected_faces:
        x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], \
        face["facial_area"]["w"], face["facial_area"]["h"]

        # Skip faces that cover the entire image
        image_area = img.shape[0] * img.shape[1]
        face_area = w * h
        if face_area > 0.9 * image_area:
            print(
                f"Skipped a face that covered the entire image in {os.path.basename(photo_path)} (likely no faces found)")
            return

        cropped_face = img[y:y + h, x:x + w]
        file_base = os.path.basename(photo_path).split('.')[0]
        face_file = f"{file_base}_face_{face_id}.jpg"
        face_path = os.path.join(faces_dir, face_file)
        cv2.imwrite(face_path, cropped_face)

        # Move the labeled face file
        label_dir = os.path.join(labeled_faces_dir, "unknown")
        os.makedirs(label_dir, exist_ok=True)

        labeled_face_path = os.path.join(label_dir, face_file)
        shutil.move(face_path, labeled_face_path)

        # Draw a bounding box and label on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),
                      2)  # Green box around the face
        cv2.putText(img, f"{face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)  # Label above the box

        # Generate SQL statement
        scaled_x, scaled_y, scaled_w, scaled_h = scale_coordinates(
            (orig_width, orig_height), (WEBSITE_WIDTH, WEBSITE_HEIGHT),
            (x, y, w, h))
        sql_statements.append(
            generate_sql(face_id, os.path.basename(photo_path), scaled_x,
                         scaled_y, scaled_w, scaled_h, "unknown"))
        face_id += 1

    # Save the annotated image with bounding boxes and labels
    annotated_path = os.path.join(annotated_photos_dir,
                                  os.path.basename(photo_path))
    cv2.imwrite(annotated_path, img)

    # Clean up
    cv2.destroyAllWindows()

    # Save SQL statements to file
    with open(SQL_FILE, 'a') as f:
        f.writelines(sql_statements)

    # Move the processed photo to the finished_photos directory
    shutil.move(photo_path,
                os.path.join(finished_photos_dir, os.path.basename(photo_path)))

    # del detected_faces
    gc.collect()


# Function to get the next available unknown label number
def get_next_unknown_number(labeled_faces_dir):
    """Get the next available 'unknown' number for labeling."""
    highest_number = 0
    pattern = re.compile(r"unknown_person_(\d{10})")
    for label in os.listdir(labeled_faces_dir):
        match = pattern.match(label)
        if match:
            number = int(match.group(1))
            if number > highest_number:
                highest_number = number

    return highest_number + 1


# Main Loop: Process all photos in the directory
for photo in os.listdir(base_dir):
    photo_path = os.path.join(base_dir, photo)
    detect_and_crop_faces(photo_path, faces_dir)
