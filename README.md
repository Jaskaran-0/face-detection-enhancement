# Veteran Identification via Facial Recognition
### Applied Research Project — Canadian Air Force Museum

A computer vision pipeline built for the **Canadian Air Force Museum** to identify WWII veterans in historical archival photographs. The system combines image restoration, automated face detection, and deep learning-based facial recognition to match degraded historical images against a reference database.

---

## Background

Archival photographs from WWII often suffer from low resolution, noise, and degradation that makes facial recognition unreliable. This project investigates whether automated photo enhancement techniques (super-resolution, brightness/contrast correction, sharpening) meaningfully improve facial recognition accuracy — or whether they are merely visual improvements with no measurable impact on model performance.

The pipeline was built for a real external client and processed a dataset of **1,000+ historical TIFF images**.

---

## Pipeline Overview

```
Raw Archival Photos (TIFF)
        ↓
Image Restoration (Real-ESRGAN super-resolution)
        ↓
Enhancement (brightness, contrast, sharpening)
        ↓
Face Detection & Cropping (OpenCV / DeepFace)
        ↓
Face Embedding Generation (InceptionResNet / ArcFace)
        ↓
Database Matching (PostgreSQL embeddings store)
        ↓
Identified Veteran + Confidence Score
```

---

## Key Features

- **Image Restoration** — Real-ESRGAN super-resolution to recover detail from degraded archival photos
- **Automated Image Slicing** — Handles large high-resolution TIFFs by intelligently slicing them to optimize GPU memory usage
- **Multiple Model Support** — Implements and compares InceptionResNet and ArcFace recognition models
- **Embedding Database** — Stores face embeddings in a SQL database for efficient similarity matching
- **Enhancement Pipeline** — Applies and tests brightness, contrast, and sharpening adjustments to evaluate their impact on recognition accuracy
- **CUDA Acceleration** — GPU-accelerated processing for large batch workloads

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.8+ |
| Recognition Models | ArcFace, InceptionResNet (via DeepFace) |
| Image Restoration | Real-ESRGAN |
| Computer Vision | OpenCV |
| Database | PostgreSQL (SQL embeddings store) |
| GPU Acceleration | CUDA |
| Scripting | PowerShell (TIFF batch processing) |

---

## Scripts

| Script | Purpose |
|---|---|
| `detect_and_label.py` | Detects faces in images, crops them, and queues for manual labeling |
| `make_db.py` | Generates face embeddings from labeled faces and stores them in the database |
| `match_archive.py` | Matches faces in the archive against the embeddings database and returns results with confidence scores |
| `reduce_filesize.ps1` | Batch resizes large TIFF files via ImageMagick to make them pipeline-ready |
| `report_directory.ps1` | Reports dimensions of all TIFF images in a directory |
| `face_labels.sql` | SQL schema for the face embeddings and label store |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Jaskaran-0/face-detection-enhancement.git
cd face-detection-enhancement
```

### 2. Set up a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install ImageMagick
Required for PowerShell TIFF batch processing. Ensure it's available in your system PATH.

---

## Usage

```bash
# Step 1 — Detect and label faces from source images
python detect_and_label.py

# Step 2 — Build the face embeddings database
python make_db.py

# Step 3 — Run matching against the full archive
python match_archive.py
```

> **Note:** The `finished_photos/` directory is not included in this repository due to the size of the original TIFF files. Add your own images locally before running the pipeline.

---

## Challenges & Technical Notes

- **CUDA compatibility** — resolved version conflicts between PyTorch, CUDA drivers, and DeepFace model backends
- **Memory constraints** — large TIFF files required automated slicing logic to prevent GPU OOM errors during batch processing
- **Model comparison** — ArcFace and InceptionResNet were both evaluated; results varied depending on image quality and enhancement method applied
- **OpenCV errors** — resolved colour space and format issues when processing aged TIFF files with non-standard encoding

---

## License

MIT License — see [LICENSE](LICENSE) for details.
