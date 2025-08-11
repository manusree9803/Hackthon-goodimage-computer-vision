
from typing import List, Optional, Tuple, Dict
import os
import math
import csv

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


# ---------------------------
# CONFIG - change paths here
# ---------------------------
GOOD_FOLDER = r"C:\Users\manus\Downloads\good"
DEFECT_FOLDER = r"C:\Users\manus\Downloads\defect"
OUTPUT_FOLDER = r"C:\Users\manus\Downloads\output_defect_results"
TARGET_SIZE = 300  # size (px) to which images are aligned
MIN_DEFECT_AREA = 1
MAX_DEFECT_AREA = 5000
RING_PAD = 5  # mask padding in px
THRESH_VALUE = 8  # threshold value for absdiff
# ---------------------------


def ensure_folders(output_root: str) -> None:
    """Create output folders if they don't exist."""
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "plots"), exist_ok=True)


def detect_ring(img_gray: np.ndarray) -> Tuple[Optional[Tuple[int, int, int]],
                                                Optional[Tuple[int, int, int]]]:
    """
    Detect outer and inner circles in a grayscale image using HoughCircles.

    Returns:
        (outer_circle, inner_circle) where each circle is (x, y, r).
        If detection fails returns (None, None).
    """
    # Adjust Hough parameters if detection fails on your dataset
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=0,
    )
    if circles is None:
        return None, None

    circles = np.round(circles[0, :]).astype(int)
    # Sort by radius (descending): largest is likely outer circle
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    outer = tuple(circles[0])
    inner = tuple(circles[-1]) if len(circles) > 1 else tuple(circles[0])
    return outer, inner


def make_ring_mask(shape: Tuple[int, int], outer: Tuple[int, int, int],
                   inner: Tuple[int, int, int], pad: int = 5) -> np.ndarray:
    """
    Create a binary mask that covers the ring region (between outer and inner).
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (outer[0], outer[1]), outer[2] + pad, 255, -1)
    cv2.circle(mask, (inner[0], inner[1]), max(inner[2] - pad, 0), 0, -1)
    return mask


def load_first_good_image(folder: str) -> Tuple[np.ndarray, np.ndarray,
                                                Tuple[int, int, int],
                                                Tuple[int, int, int]]:
    """
    Load first image from good folder, return color, gray and detected rings.
    Raises ValueError on failure.
    """
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        raise ValueError("No image found in good folder.")
    img_path = os.path.join(folder, files[0])
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outer, inner = detect_ring(gray)
    if outer is None or inner is None:
        raise ValueError("Could not detect circles in good image.")
    return img, gray, outer, inner


def resize_and_center(img_gray: np.ndarray, img_color: np.ndarray,
                      outer_circle: Tuple[int, int, int],
                      target_size: int = TARGET_SIZE) -> Tuple[np.ndarray, np.ndarray,
                                                              Tuple[int, int, int]]:
    """
    Resize image based on outer circle radius and center the outer circle
    at (target_size//2, target_size//2).
    Returns resized_gray, resized_color, new_outer_circle
    """
    scale = target_size / (outer_circle[2] * 2)
    resized_gray = cv2.resize(img_gray, None, fx=scale, fy=scale)
    resized_color = cv2.resize(img_color, None, fx=scale, fy=scale)
    outer_new, _ = detect_ring(resized_gray)
    if outer_new is None:
        # fallback: assume center at image center with scaled radius
        h, w = resized_gray.shape
        outer_new = (w // 2, h // 2, int(outer_circle[2] * scale))
    # compute shift so outer circle center is at target center
    shift_x = target_size // 2 - outer_new[0]
    shift_y = target_size // 2 - outer_new[1]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_gray = cv2.warpAffine(resized_gray, M, (target_size, target_size))
    aligned_color = cv2.warpAffine(resized_color, M, (target_size, target_size))
    outer_aligned = (target_size // 2, target_size // 2, outer_new[2])
    return aligned_gray, aligned_color, outer_aligned


def compare_and_find_defects(good_masked: np.ndarray,
                             defect_masked: np.ndarray,
                             thresh_value: int = THRESH_VALUE) -> Tuple[np.ndarray, float]:
    """
    Compare masked good and defect images using SSIM + absdiff.
    Returns (binary_threshold_image, ssim_score)
    """
    score, _ = ssim(good_masked, defect_masked, full=True)
    diff = cv2.absdiff(good_masked, defect_masked)
    _, thresh = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY)
    # small open + dilation to remove speckle noise and connect nearby pixels
    kernel_open = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    kernel_dilate = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=2)
    return thresh, float(score)


def localize_and_classify(thresh: np.ndarray, display_img: np.ndarray,
                          outer_radius: int, inner_radius: int,
                          min_area: int = MIN_DEFECT_AREA,
                          max_area: int = MAX_DEFECT_AREA) -> Tuple[List[Tuple[int, int, int, int]],
                                                                  str]:
    """
    Find contours on threshold image, localize defects (bounding boxes),
    and classify by distance from center.
    Returns (list_of_bboxes, classification_string).
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    defect_positions: List[Tuple[int, int, int, int]] = []
    flash_found = False
    cut_found = False
    center = (TARGET_SIZE // 2, TARGET_SIZE // 2)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        defect_positions.append((x, y, w, h))
        # draw bounding box on display image for visualization
        cv2.rectangle(display_img, (x, y), (x + w, y + h),
                      (0, 255, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        dist = math.hypot(cx - center[0], cy - center[1])
        # classification thresholds use ring radii with small margins
        if dist >= outer_radius - 5:
            flash_found = True
        elif dist <= inner_radius + 5:
            cut_found = True

    if flash_found and cut_found:
        classification = "mixed_defects"
    elif flash_found:
        classification = "defect_flashes"
    elif cut_found:
        classification = "defect_cut_marks"
    else:
        classification = "good_like"

    return defect_positions, classification


def save_summary_csv(results: List[Dict], output_path: str) -> None:
    """Save results list of dicts to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def plot_and_save_distribution(results: List[Dict], output_root: str) -> None:
    """Create simple bar & pie charts for defect distribution and save."""
    types = [r["classification"] for r in results]
    counts = {t: types.count(t) for t in set(types)}
    # bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("Defect Type Distribution")
    plt.xlabel("Defect Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "plots", "defect_distribution_bar.png"))
    plt.close()

    # pie chart
    plt.figure(figsize=(5, 5))
    plt.pie(list(counts.values()), labels=list(counts.keys()),
            autopct="%1.1f%%")
    plt.title("Defect Type Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "plots", "defect_distribution_pie.png"))
    plt.close()


def process_all_defects(good_gray: np.ndarray, good_color: np.ndarray,
                        outer_good: Tuple[int, int, int],
                        inner_good: Tuple[int, int, int],
                        defect_folder: str,
                        output_root: str) -> List[Dict]:
    """Main loop to process all defect images and return results list."""
    results: List[Dict] = []
    for fname in os.listdir(defect_folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        file_path = os.path.join(defect_folder, fname)
        defect_color = cv2.imread(file_path)
        if defect_color is None:
            print(f"Warning: failed to read {file_path}, skipping.")
            continue
        defect_gray = cv2.cvtColor(defect_color, cv2.COLOR_BGR2GRAY)

        outer_def, inner_def = detect_ring(defect_gray)
        if outer_def is None or inner_def is None:
            msg = f"Skipping {fname}: no circles detected"
            print(msg)
            results.append({
                "filename": fname,
                "classification": "undetected_ring",
                "ssim_score": None,
                "defect_count": 0,
                "defect_positions": [],
                "remarks": msg
            })
            continue

        defect_gray_aligned, defect_color_aligned, outer_def_aligned = \
            resize_and_center(defect_gray, defect_color, outer_def,
                              target_size=TARGET_SIZE)

        # Prepare aligned good image (pad/crop if needed)
        good_gray_aligned, good_color_aligned, outer_good_aligned = \
            resize_and_center(good_gray, good_color, outer_good,
                              target_size=TARGET_SIZE)

        # Create ring mask using aligned outer/inner radii (centered)
        mask = make_ring_mask(good_gray_aligned.shape,
                              (TARGET_SIZE // 2, TARGET_SIZE // 2,
                               outer_good_aligned[2]),
                              (TARGET_SIZE // 2, TARGET_SIZE // 2,
                               inner_good[2]),
                              pad=RING_PAD)
        good_masked = cv2.bitwise_and(good_gray_aligned,
                                      good_gray_aligned, mask=mask)
        defect_masked = cv2.bitwise_and(defect_gray_aligned,
                                        defect_gray_aligned, mask=mask)

        thresh_img, score = compare_and_find_defects(good_masked,
                                                     defect_masked,
                                                     thresh_value=THRESH_VALUE)

        display_img = defect_color_aligned.copy()
        defect_positions, classification = localize_and_classify(
            thresh_img, display_img, outer_good_aligned[2], inner_good[2]
        )

        # Save visualization and results
        out_image_path = os.path.join(output_root, "images", fname)
        cv2.imwrite(out_image_path, display_img)

        results.append({
            "filename": fname,
            "classification": classification,
            "ssim_score": round(score, 4) if score is not None else None,
            "defect_count": len(defect_positions),
            "defect_positions": defect_positions,
            "remarks": ""
        })

    return results


def main():
    """Main entry point."""
    ensure_folders(OUTPUT_FOLDER)
    try:
        good_color, good_gray, outer_good, inner_good = \
            load_first_good_image(GOOD_FOLDER)
    except ValueError as exc:
        print(f"Error loading good image: {exc}")
        return

    results = process_all_defects(good_gray, good_color, outer_good,
                                  inner_good, DEFECT_FOLDER, OUTPUT_FOLDER)

    csv_path = os.path.join(OUTPUT_FOLDER, "results_summary.csv")
    save_summary_csv(results, csv_path)
    print(f"Saved CSV results to: {csv_path}")

    # Simple visualizations
    plot_and_save_distribution(results, OUTPUT_FOLDER)
    print("Saved distribution plots to output/plots/")

    print("Processing complete.")


if __name__ == "__main__":
    main()
