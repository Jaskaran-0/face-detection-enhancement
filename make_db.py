import os
import pickle
from deepface import DeepFace

# Define paths and model name
labeled_faces_dir = './labeled_faces'
faces_db_path = './faces_database.pkl'
#model_name = "VGG-Face"
model_name = "Facenet"

# Create faces database
faces_database = []

for person_name in os.listdir(labeled_faces_dir):
    person_dir = os.path.join(labeled_faces_dir, person_name)
    
    if not os.path.isdir(person_dir):
        continue

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        try:
            representation = DeepFace.represent(img_path=image_path, model_name=model_name)[0]["embedding"]
            face_data = {
                'person': person_name,
                'embedding': representation,
                'image_path': image_path
            }
            faces_database.append(face_data)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Save the faces database to a pickle file
with open(faces_db_path, 'wb') as f:
    pickle.dump(faces_database, f)

print("Faces database created and saved successfully.")
