import os
import re
from deepface import DeepFace

# Define paths and model name
archive_dir = './photos1'
faces_db_path = './labeled_faces1'
model_name = "Facenet"
detector_backend = "mtcnn"
distance_metric = "cosine"

# Traverse the archive directory and process images
for root, dirs, files in os.walk(archive_dir):
    for file_name in files:
        # Skip files with '-Back' in their names
        if re.search(r'-Back', file_name, re.IGNORECASE):
            continue
        
        image_path = os.path.join(root, file_name)
        try:
            # Perform face matching
            dfs = DeepFace.find(img_path=image_path, db_path=faces_db_path, model_name=model_name,
                                distance_metric=distance_metric, detector_backend=detector_backend)
            for df in dfs:
                if not df.empty:
                    for index, row in df.iterrows():
                        # Find the correct column for the distance metric
                        distance_column = next((col for col in df.columns if col.endswith('_cosine')), None)
                        if distance_column:
                            print(f"Match found for {file_name} in {root}: {row['identity']}, distance: {row[distance_column]}")
                        else:
                            print(f"No distance column found for {file_name} in {root}")
                else:
                    print(f"No match found for {file_name} in {root}")
        except Exception as e:
            # Capture and print detailed exception information
            print(f"Error processing {image_path}: {str(e)}")

print("Face matching completed.")
