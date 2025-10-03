import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import matplotlib.pyplot as plt

def main(image_path, model_path):
    # Resolve absolute paths for debugging
    image_path = os.path.abspath(image_path)
    model_path = os.path.abspath(model_path)

    print("🔍 Image path:", image_path)
    print("🔧 Model path:", model_path)

    # Check if files exist
    if not os.path.exists(image_path):
        print("❌ Error: Image file not found")
        return
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found")
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not load image (corrupted or unsupported format)")
        return

    # Convert BGR (OpenCV) to RGB (for display)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    img_name = os.path.basename(image_path)

    # Load model and predict
    try:
        model = YOLO(model_path)
        results = model(image_rgb, save=False)  # Use RGB for consistency
    except Exception as e:
        print(f"❌ Error during model inference: {e}")
        return

    # Check for masks
    if not results[0].masks:
        # No tumor detected
        report = (
            f"Medical Image Analysis Report\n"
            f"Image: {img_name}\n"
            f"Finding: No brain tumor region detected by the segmentation model.\n"
            f"Confidence: N/A\n"
            f"Note: Absence of detection does not rule out pathology. Model accuracy may vary.\n"
            f"Processed on: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        print(report)

        # Show original image
        plt.figure(figsize=(8, 8))
        plt.imshow(image_rgb)
        plt.title("No Tumor Detected", fontsize=16, color='green')
        plt.axis("off")
        plt.show()

        return report

    # Get all masks
    masks = results[0].masks.xy
    num_regions = len(masks)

    # Compute total affected area
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_points in masks:
        mask_points = mask_points.astype(np.int32)
        cv2.fillPoly(binary_mask, [mask_points], 1)
    total_pixels = np.sum(binary_mask)
    area_percentage = (total_pixels / (h * w)) * 100

    # Estimate rough location (center of mass)
    moments = cv2.moments(binary_mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        relative_x = cx / w
        relative_y = cy / h

        x_label = "left side" if relative_x < 0.33 else "right side" if relative_x > 0.66 else "central region"
        y_label = "upper part" if relative_y < 0.33 else "lower part" if relative_y > 0.66 else "middle part"
        location = f"{y_label}, {x_label}"
    else:
        location = "center could not be computed"

    # Average confidence
    confidences = results[0].boxes.conf.cpu().numpy()
    avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else "N/A"

    # Generate LLM-friendly RAG input
    report = (
        f"Brain Tumor Segmentation Analysis\n"
        f"Source Image: {img_name}\n"
        f"Detection Status: Tumor region detected\n"
        f"Number of Lesions: {num_regions}\n"
        f"Total Affected Area: {total_pixels} pixels ({area_percentage:.2f}% of brain area)\n"
        f"Approximate Location: {location}\n"
        f"Model Confidence (avg): {avg_conf:.3f}\n"
        f"Segmentation Method: YOLOv11 Segmentation model\n"
        f"Processing Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Note: This is an automated analysis. Clinical validation is required."
    )

    # Create overlay for visualization
    overlay = image.copy()
    for mask_points in masks:
        pts = mask_points.astype(np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green fill
        cv2.polylines(overlay, [pts], True, (0, 0, 255), thickness=2)  # Red contour

    alpha = 0.6
    segmented_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    segmented_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    # Save segmented image
    output_img_name = f"segmented_{os.path.splitext(img_name)[0]}.jpg"
    cv2.imwrite(output_img_name, segmented_image)
    print(f"✅ Segmented visualization saved as {output_img_name}")

    # Display images using matplotlib
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Brain MRI", fontsize=14)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title(f"Segmented Tumor (Area: {area_percentage:.1f}%)", fontsize=14, color='red')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Print RAG-ready report
    print("\n" + "="*60)
    print("📄 RAG-READY MEDICAL REPORT (FOR LLM INPUT)")
    print("="*60)
    print(report)
    print("="*60)

    return report

# ————————————————————————
# Run the function
# ————————————————————————

if __name__ == "__main__":
    # ✅ Update these paths based on your setup
    default_image_path = r"C:\Users\Bilal\Desktop\BrainTumorVA\BRAIN-TUMOR-1\test\images\y2_jpg.rf.f92efc4de05a488452943801fa495bdc.jpg"
    default_model_path = r"C:\Users\Bilal\Desktop\BrainTumorVA\best.pt"

    # If using Drive, uncomment and adjust:
    # default_image_path = "/content/drive/MyDrive/data/test/images/example.jpg"
    # default_model_path = "/content/drive/MyDrive/runs/segment/train/weights/best.pt"

    main(default_image_path, default_model_path)