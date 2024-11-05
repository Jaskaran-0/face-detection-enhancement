# Applied Research Project

## Project Overview
This project investigates whether photo enhancement techniques can improve the performance of facial recognition systems or if the enhancements are merely visual improvements without significant impact on recognition accuracy. The project uses Python for image processing and DeepFace for facial recognition.

## Directory Structure
The project structure includes the following key directories:
- `archive/`: Contains archived images after processing.
- `faces/`: Stores detected faces from the images.
- `finished_photos/`: Contains the original set of TIFF images (not included in GitHub repository).
- `labeled_faces/`: Contains manually labeled faces for recognition training.
- `photos/`: Stores original and processed images.

Key scripts include:
- `detect_and_label.py`: Detects faces in images, crops them, and labels them.
- `make_db.py`: Creates a face embeddings database from the labeled faces.
- `match_archive.py`: Matches faces in the `finished_photos` directory against the database.
- `reduce_filesize.ps1`: Resizes large TIFF files to make them more manageable.
- `report_directory.ps1`: Reports dimensions of TIFF images in a directory.

## Prerequisites
The project requires the following software:
- Python 3.8 or above
- Git
- ImageMagick (for resizing TIFF images in PowerShell scripts)

## Installation Instructions

### Step 1: Clone the Repository
To clone the project repository, run the following command:
```bash
git clone https://github.com/Jaskaran-0/face-detection-enhancement.git
```

Navigate into the project directory:
```bash
cd face-detection-enhancement
```

### Step 2: Set Up Virtual Environment(optional)
It is recommended to create a virtual environment to manage the project dependencies.

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Once the virtual environment is activated, install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 4: ImageMagick Installation
Ensure that ImageMagick is installed and available in your system's PATH. This is necessary for the PowerShell scripts to function correctly.

### Step 5: Running the Scripts
- **Face Detection and Labeling**: Use `detect_and_label.py` to detect and manually label faces from the images.
  ```bash
  python detect_and_label.py
  ```
- **Create Face Embeddings Database**: Run `make_db.py` to create a database of face embeddings.
  ```bash
  python make_db.py
  ```
- **Match Faces in Archive**: Use `match_archive.py` to match faces from the `finished_photos` directory with those in the embeddings database.
  ```bash
  python match_archive.py
  ```

## Important Notes
- The `finished_photos/` directory is not included in the repository due to the large size of the TIFF images. Make sure to add your own images to this folder locally.
- Use the `.gitignore` file to exclude large folders and generated data from the repository.

## Contributing
Feel free to create branches and work on different features. Always ensure to **pull the latest changes** before starting your work to avoid merge conflicts.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
