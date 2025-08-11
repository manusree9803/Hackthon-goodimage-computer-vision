# Defect Detection and Classification in Images

This project detects and classifies defects on circular objects (e.g., rings) by comparing images of "good" (defect-free) and "defect" samples. It aligns and analyzes images, detects defects, classifies defect types, and generates summary reports and visualizations.

---

## Features

- Automatically detects circular regions (rings) in images using Hough Circle Transform.
- Aligns and resizes images to a standard size for consistent comparison.
- Compares "good" and "defect" images using Structural Similarity Index (SSIM) and pixel difference.
- Detects defect regions and classifies defects into categories:
  - `defect_flashes`
  - `defect_cut_marks`
  - `mixed_defects`
  - `good_like`
  - `undetected_ring` (if ring detection fails)
- Generates bounding box visualizations on defect images.
- Saves results in CSV format and generates bar and pie charts showing defect distribution.
- Modular, well-commented Python code following PEP8 guidelines.

---

## Project Structure

project_root/
│
├── good/ # Folder containing defect-free images
├── defect/ # Folder containing defect images
├── output_defect_results/ # Output folder for results, images, plots, and CSV
│ ├── images/
│ └── plots/
├── defect_detection.py # Main Python script with defect detection logic
└── README.md1 # This file



---

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- pandas
- matplotlib
- scikit-image

Install dependencies using:

```bash
pip install opencv-python numpy pandas matplotlib scikit-image
