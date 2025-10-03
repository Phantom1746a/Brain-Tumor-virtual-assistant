# yolo_segmentation.py
import cv2
import numpy as np
from ultralytics import YOLO
import os
from config import YOLO_REPORT_PATH
import tempfile

def run_segmentation(input_img_path, model_path="best.pt"):
    """
    Runs YOLO segmentation on input image (path or in-memory).
    Returns path to segmented image or "no_tumor" or None.
    """
    # Read image
    image = cv2.imread(input_img_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    img_name = os.path.basename(input_img_path)

    try:
        model = YOLO(model_path)
        results = model(image, save=False)
    except Exception as e:
        print("ModelError:", e)
        return None

    # No masks
    if not results[0].masks:
        report = (
            f"Medical Image Analysis Report\n"
            f"Image: {img_name}\n"
            f"Finding: No tumor detected.\n"
            f"Note: AI-based analysis. Consult radiologist.\n"
            f"Timestamp: {os.popen('date').read().strip() if os.name != 'nt' else ''}"
        )
        os.makedirs(os.path.dirname(YOLO_REPORT_PATH), exist_ok=True)
        with open(YOLO_REPORT_PATH, "w") as f:
            f.write(report)
        return "no_tumor"

    # Process masks
    masks = results[0].masks.xy
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    for pts in masks:
        cv2.fillPoly(binary_mask, [pts.astype(np.int32)], 1)
    area_pixels = np.sum(binary_mask)
    area_percent = (area_pixels / (h * w)) * 100

    # Location
    m = cv2.moments(binary_mask)
    location = "unknown"
    if m["m00"] > 0:
        cx, cy = int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])
        x_loc = "left" if cx < w*0.33 else "right" if cx > w*0.66 else "central"
        y_loc = "upper" if cy < h*0.33 else "lower" if cy > h*0.66 else "middle"
        location = f"{y_loc}, {x_loc}"

    # Confidence
    conf = results[0].boxes.conf.cpu().numpy()
    avg_conf = float(np.mean(conf)) if len(conf) > 0 else "N/A"

    # Save report
    report = (
        f"Brain Tumor Segmentation Analysis\n"
        f"Source Image: {img_name}\n"
        f"Detection: Tumor detected\n"
        f"Lesions: {len(masks)}\n"
        f"Area: {area_pixels} px ({area_percent:.2f}%)\n"
        f"Location: {location}\n"
        f"Confidence: {avg_conf:.3f}\n"
        f"Method: YOLOv8 Segmentation\n"
        f"Timestamp: {os.popen('date').read().strip() if os.name != 'nt' else ''}"
    )
    os.makedirs(os.path.dirname(YOLO_REPORT_PATH), exist_ok=True)
    with open(YOLO_REPORT_PATH, "w") as f:
        f.write(report)

    # Save segmented image to temp file
    overlay = image.copy()
    for pts in masks:
        p = pts.astype(np.int32)
        cv2.fillPoly(overlay, [p], (0, 255, 0))
        cv2.polylines(overlay, [p], True, (0, 0, 255), 2)
    segmented = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # Use temp file
    temp_seg = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_seg.name, segmented)
    return temp_seg.name  # Return path to segmented image