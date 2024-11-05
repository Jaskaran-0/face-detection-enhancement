# Tested 2024-07-08 @ 1224
# Working as expected, sensitive to memory error.

# Options selected from research paper https://dergipark.org.tr/en/pub/gazibtd/issue/84331/1399077

# Face Detection Backend Options {Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).}
BACKEND = 'retinaface'
#BACKEND = 'centerface' # reported to outperform retinaface
# Face Recognition Model Options {FaceNet-128d, FaceNet512d, VGG-Face, ArcFace, Dlib, SFace, OpenFace, DeepFace, DeepId}
FACEREC = 'FaceNet512'
# use cosine-distance or euclidean distance
# use alignment = true

# Define directories
base_dir = "photos"
faces_dir = "faces"
labeled_faces_dir = "labeled_faces"
finished_photos_dir = "finished_photos"

import os
import cv2
from deepface import DeepFace
import shutil
import re
import gc

# Constants for maximum window size
MAX_WINDOW_WIDTH = 1200
MAX_WINDOW_HEIGHT = 800
WINDOW_POSITION_X = 100
WINDOW_POSITION_Y = 100

# Constants for website image dimensions
WEBSITE_WIDTH = 1800
WEBSITE_HEIGHT = 800
SQL_FILE = "face_labels.sql"
TABLE_NAME = "face_bounding_boxes"

# Constant for preview box size
PREVIEW_BOX_SIZE = 500

os.makedirs(faces_dir, exist_ok=True)
os.makedirs(labeled_faces_dir, exist_ok=True)
os.makedirs(finished_photos_dir, exist_ok=True)

def resize_image(image, max_width=800):
    height, width = image.shape[:3]
    if width > max_width:
        scaling_factor = max_width / width
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def resize_for_preview(image, box_size=PREVIEW_BOX_SIZE):
    height, width = image.shape[:2]
    if height > box_size or width > box_size:
        scaling_factor = min(box_size / height, box_size / width)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def resize_to_fit_window(image, max_width=MAX_WINDOW_WIDTH, max_height=MAX_WINDOW_HEIGHT):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def scale_coordinates(orig_dim, new_dim, coord):
    scale_x = new_dim[0] / orig_dim[0]
    scale_y = new_dim[1] / orig_dim[1]
    return int(coord[0] * scale_x), int(coord[1] * scale_y), int(coord[2] * scale_x), int(coord[3] * scale_y)

def generate_sql(face_id, photo_name, x, y, w, h, label):
    return f"INSERT INTO {TABLE_NAME} (face_id, photo_name, x, y, width, height, label) VALUES ('{face_id}', '{photo_name}', {x}, {y}, {w}, {h}, '{label}');\n"

def detect_and_crop_faces(photo_path, faces_dir):
    print(f"Processing {photo_path}")
    img = cv2.imread(photo_path)
    orig_height, orig_width = img.shape[:2]
    detected_faces = DeepFace.extract_faces(img, detector_backend=BACKEND, align=True, enforce_detection=False)
    print(f"Total faces detected in {os.path.basename(photo_path)}: {len(detected_faces)}")
    
    sql_statements = []
    face_id = 1
    unknown_counter = get_next_unknown_number(labeled_faces_dir)

    # Order faces from top to bottom, left to right
    detected_faces.sort(key=lambda face: (face["facial_area"]["y"], face["facial_area"]["x"]))
    
    # Scale the entire image to website size
    scaled_img = cv2.resize(img, (WEBSITE_WIDTH, WEBSITE_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # Label each face
    for face in detected_faces:
        x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]

        # Skip faces that cover the entire image
        image_area = img.shape[0] * img.shape[1]
        face_area = w * h
        if face_area > 0.9 * image_area:
            print(f"Skipped a face that covered the entire image in {os.path.basename(photo_path)} (likely no faces found)")
            return

        cropped_face = img[y:y+h, x:x+w]
        file_base = os.path.basename(photo_path).split('.')[0]
        face_file = f"{file_base}_face_{face_id}.jpg"
        face_path = os.path.join(faces_dir, face_file)
        cv2.imwrite(face_path, cropped_face)

        # Draw bounding box for the current face
        scaled_img_with_box = scaled_img.copy()
        scaled_x, scaled_y, scaled_w, scaled_h = scale_coordinates((orig_width, orig_height), (WEBSITE_WIDTH, WEBSITE_HEIGHT), (x, y, w, h))
        cv2.rectangle(scaled_img_with_box, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 2)

        # Display the image with the bounding box
        frame_name = "Image with Faces" + file_base
        cv2.imshow(frame_name, scaled_img_with_box)
        cv2.waitKey(1)
        cv2.moveWindow(frame_name, WINDOW_POSITION_X, WINDOW_POSITION_Y)
        
        label = input(f"Enter label for face {face_id}: ").strip()
        if not label:
            label = f"unknown_person_{unknown_counter:010d}"
            unknown_counter += 1
        
        # Move the labeled face file
        label_dir = os.path.join(labeled_faces_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        labeled_face_path = os.path.join(label_dir, face_file)
        shutil.move(face_path, labeled_face_path)
        
        # Generate SQL statement
        scaled_x, scaled_y, scaled_w, scaled_h = scale_coordinates((orig_width, orig_height), (WEBSITE_WIDTH, WEBSITE_HEIGHT), (x, y, w, h))
        sql_statements.append(generate_sql(face_id, os.path.basename(photo_path), scaled_x, scaled_y, scaled_w, scaled_h, label))
        face_id += 1
    
    cv2.destroyAllWindows()        

    # Save SQL statements to file
    with open(SQL_FILE, 'a') as f:
        f.writelines(sql_statements)

    # Move the processed photo to the finished_photos directory
    shutil.move(photo_path, os.path.join(finished_photos_dir, os.path.basename(photo_path)))

    # Free memory
    del img
    del detected_faces
    gc.collect()

def get_next_unknown_number(labeled_faces_dir):
    highest_number = 0
    pattern = re.compile(r"unknown_person_(\d{10})")
    
    for label in os.listdir(labeled_faces_dir):
        match = pattern.match(label)
        if match:
            number = int(match.group(1))
            if number > highest_number:
                highest_number = number

    return highest_number + 1

def manual_label_faces(faces_dir, labeled_faces_dir):
    unknown_counter = get_next_unknown_number(labeled_faces_dir)

    for face_file in os.listdir(faces_dir):
        face_path = os.path.join(faces_dir, face_file)
        img = cv2.imread(face_path)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        
        label = input("Enter label for this face: ").strip()
        if not label:
            label = f"unknown_person_{unknown_counter:010d}"
            unknown_counter += 1
        
        label_dir = os.path.join(labeled_faces_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        labeled_face_path = os.path.join(label_dir, face_file)
        shutil.move(face_path, labeled_face_path)
        cv2.destroyAllWindows()

    # Free memory
    del img
    gc.collect()

# Loop through all photos in the base directory
for photo in os.listdir(base_dir):
    photo_path = os.path.join(base_dir, photo)
    detect_and_crop_faces(photo_path, faces_dir)