import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer
from realesrgan import RealESRGANer as RealESRGAN
from realesrgan.models.realesrgan_model import RealESRGANModel

# Directories setup
base_dir = "photos1"
labeled_faces_dir = "labeled_faces1"
annotated_photos_dir = "annotated_photos"
weights_dir = "weights"

# Create directories if they don't exist
os.makedirs(labeled_faces_dir, exist_ok=True)
os.makedirs(annotated_photos_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# Initialize device for GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Define paths to model weights
gfpgan_model_path = os.path.join(weights_dir, 'GFPGANv1.3.pth')
realesrgan_model_path = os.path.join(weights_dir, 'RealESRGAN_x4plus.pth')

# Download the model weights if they are not already available
if not os.path.exists(gfpgan_model_path):
    print("Downloading GFPGAN model weights...")
    gfpgan_model_path = load_file_from_url(
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.4/GFPGANv1.4.pth",
        model_dir=weights_dir,
        progress=True
    )

# if not os.path.exists(realesrgan_model_path):
#     print("Downloading RealESRGAN model weights...")
#     realesrgan_model_path = load_file_from_url(
#         url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
#         model_dir=weights_dir,
#         progress=True
#     )

# Initialize GFPGAN for face restoration
restorer = GFPGANer(
    model_path=gfpgan_model_path,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device=device
)
#
# opt = {
#     'network_g': {
#         'type': 'RRDBNet',
#         'num_in_ch': 3,
#         'num_out_ch': 3,
#         'scale': 4
#     },
#     'scale': 4,
#     'is_train': False,
# 'num_gpu': 1,
# 'dist': False
# }
# model = RealESRGANModel(opt= opt)
# upsampler = RealESRGAN(
#     scale=2,
#     model_path=realesrgan_model_path,
#     model=model,
#     tile=0,
#     tile_pad=10,
#     pre_pad=10,
#     half=False,
#     device=device
# )

# Define maximum target file size in bytes (20MB)
MAX_FILE_SIZE = 20 * 1024 * 1024


def resize_image_opencv(image, target_size=MAX_FILE_SIZE):
    """
    Resize the image while keeping the quality and maintaining a file size below the target.
    """
    encode_param = [cv2.IMWRITE_TIFF_COMPRESSION, 1]  # Lossless compression for TIFF

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Error: Unable to load the image. Please check the file path or file format.")

    # Get the current size of the encoded image
    current_size = len(cv2.imencode('.tif', image, encode_param)[1].tobytes())

    # Reduce the size by scaling down while it's larger than the target size
    while current_size > target_size:
        # Calculate new dimensions based on scale factor
        height, width = image.shape[:2]
        scaling_factor = 0.9
        new_size = (int(width * scaling_factor), int(height * scaling_factor))

        # Resize the image
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

        # Update current file size
        current_size = len(cv2.imencode('.tif', image, encode_param)[1].tobytes())

    return image

# def upscale_image(image):
#     """
#     Use Real-ESRGAN to upscale an image.
#     """
#     sr_image, _ = upsampler.enhance(image, outscale=4)
#     return sr_image

def restore_faces(image):
    """
    Use GFPGAN to restore the quality of faces in an image.
    """
    _, restored_image, _ = restorer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_image

def detect_and_crop_faces(photo_path, labeled_faces_dir, annotated_photos_dir):
    """
    Detect faces in the image, annotate them, and save the cropped face images and annotated photos.
    """
    # Read the image using OpenCV
    img = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Warning: Unable to load image {photo_path}. Skipping...")
        return

    # If the image has 4 channels (e.g., RGBA), convert it to 3 channels (RGB)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Resize image to fit within the 20MB limit
    img = resize_image_opencv(img)

    # Upscale the image using Real-ESRGAN
    # img = upscale_image(img)

    # Restore face quality using GFPGAN
    img = restore_faces(img)

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(img)

    # If no faces are detected, skip the image
    if boxes is None:
        print(f"No faces detected in {photo_path}.")
        return

    # Annotate the image with bounding boxes
    for i, box in enumerate(boxes):
        # box = box[0] if len(
        #     box.shape) > 1 else box  # Extract values if extra dimensions

        x1, y1, x2, y2 = [int(coord) for coord in box]
        # Draw a rectangle around the face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add a label to the bounding box
        cv2.putText(img, f'Face {i + 1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Crop the face from the original image
        cropped_face = img[y1:y2, x1:x2]

        # Save the cropped face image
        base_filename = os.path.splitext(os.path.basename(photo_path))[0]
        face_folder = os.path.join(labeled_faces_dir, base_filename)
        os.makedirs(face_folder, exist_ok=True)
        face_filename = f"{base_filename}_face_{i + 1}.tif"
        face_path = os.path.join(face_folder, face_filename)
        cv2.imwrite(face_path, cropped_face, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    # Save the annotated image
    annotated_image_path = os.path.join(annotated_photos_dir, os.path.basename(photo_path))
    cv2.imwrite(annotated_image_path, img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    print(f"Processed {photo_path}: {len(boxes)} face(s) detected and saved.")


# Process each image in the base directory
for photo in os.listdir(base_dir):
    if photo.lower().endswith(".tif"):
        photo_path = os.path.join(base_dir, photo)
        detect_and_crop_faces(photo_path, labeled_faces_dir, annotated_photos_dir)
