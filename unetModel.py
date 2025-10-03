import numpy as np
import tensorflow as tf
import nibabel as nib
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "best_unet_brats.h5"  # Your trained 2D U-Net model
PATIENT_ID = "BraTS20_Validation_088"
DATA_DIR = r"C:\Users\Bilal\Desktop\BrainTumorVA\dataset\BraTS20_Validation_088"

# Use .nii extension
MODALITIES = {
    "flair": os.path.join(DATA_DIR, f"{PATIENT_ID}_flair.nii"),
    "t1": os.path.join(DATA_DIR, f"{PATIENT_ID}_t1.nii"),
    "t1ce": os.path.join(DATA_DIR, f"{PATIENT_ID}_t1ce.nii"),
    "t2": os.path.join(DATA_DIR, f"{PATIENT_ID}_t2.nii")
}

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("💡 Ensure 'best_unet_brats.h5' is in the correct folder.")
    sys.exit(1)

# Check if all 4 modalities exist
missing = []
for name, path in MODALITIES.items():
    if not os.path.exists(path):
        missing.append(os.path.basename(path))
if missing:
    print(f"❌ Missing MRI files: {missing}")
    print("💡 You need all 4 modalities: FLAIR, T1, T1ce, T2")
    print("📥 BraTS 2020 dataset: https://www.med.upenn.edu/cbica/brats2020/")
    sys.exit(1)

# Segment classes (BraTS)
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

# ================================
# Load Model
# ================================
print("🧠 Loading 2D U-Net model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Loaded model: {MODEL_PATH}")
    print(f"🧩 Expected input shape: {model.input_shape}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Verify input shape
expected_input_shape = (None, 240, 240, 4)
if model.input_shape != expected_input_shape:
    print(f"❌ Model input shape {model.input_shape} is not compatible with (None, 240, 240, 4)")
    print("🔧 Train a 2D U-Net model with input shape (240, 240, 4)")
    sys.exit(1)

# ================================
# Load and Preprocess All 4 Modalities
# ================================
print(f"📁 Loading 4 MRI modalities for {PATIENT_ID}...")
data_vol = []
for mod_name, mod_path in MODALITIES.items():
    print(f"  → Loading {mod_name}...")
    img = nib.load(mod_path)
    vol = img.get_fdata()  # Shape: (240, 240, 155)

    # Ensure correct shape
    target_shape = (240, 240, 155)
    if vol.shape != target_shape:
        print(f"⚠️  {mod_name} has shape {vol.shape}, expected {target_shape}. Skipping.")
        sys.exit(1)

    # Normalize: min-max to [0,1] on non-zero voxels (consistent with training)
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-6)
    data_vol.append(vol)

# Stack into 4D volume: (240, 240, 155, 4)
mri_volume = np.stack(data_vol, axis=-1).astype(np.float32)
print(f"✅ Final MRI volume shape: {mri_volume.shape}")

# ================================
# Run 2D Slice-Wise Inference
# ================================
print("🚀 Running 2D U-Net Inference...")
pred_mask_3d = np.zeros((240, 240, 155), dtype=np.uint8)  # To store 3D mask

# Process each axial slice
for z in range(mri_volume.shape[2]):
    # Extract 2D slice: (240, 240, 4)
    slice_2d = mri_volume[:, :, z, :]  # Shape: (240, 240, 4)
    slice_2d = np.expand_dims(slice_2d, axis=0)  # Add batch dim: (1, 240, 240, 4)

    # Predict
    pred = model.predict(slice_2d, verbose=0)  # Shape: (1, 240, 240, 4)
    pred_mask_2d = np.argmax(pred[0], axis=-1).astype(np.uint8)  # Shape: (240, 240)

    # Store in 3D mask
    pred_mask_3d[:, :, z] = pred_mask_2d
    print("For loop step",z,"/ 154")
print("✅ Inference completed.")

# ================================
# Analyze Results
# ================================
print("\n📊 Tumor Segmentation Results")
print("=" * 60)

total_voxels = pred_mask_3d.size
core = np.sum(pred_mask_3d == 1)
edema = np.sum(pred_mask_3d == 2)
enhancing = np.sum(pred_mask_3d == 3)
tumor_volume = core + edema + enhancing
percentage = (tumor_volume / total_voxels) * 100

print(f"🧩 Total Brain Voxels:     {total_voxels:,}")
print(f"🔴 Necrotic/Core:          {core:,} voxels")
print(f"🟡 Edema:                  {edema:,} voxels")
print(f"🟢 Enhancing:              {enhancing:,} voxels")
print(f"💥 Total Tumor Volume:     {tumor_volume:,} voxels ({percentage:.2f}%)")

if tumor_volume == 0:
    print("🟢 No tumor detected.")
else:
    xs, ys, zs = np.where(pred_mask_3d > 0)
    cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)
    print(f"📍 Tumor Center (approx):  X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f}")

    x_loc = "Left" if cx < 120 else "Right"
    y_loc = "Front" if cy < 120 else "Back"
    z_loc = "Bottom" if cz < 77 else "Middle" if cz < 116 else "Top"
    print(f"📍 Anatomical Location:    {y_loc}-{x_loc}, {z_loc} brain")

print("=" * 60)
print("✅ Inference complete. Results printed to terminal.")

# ================================
# Visualize Results
# ================================
print("🖼️ Generating visualization...")

# Choose middle slice
mid_z = pred_mask_3d.shape[2] // 2
mri_slice = mri_volume[:, :, mid_z, 0]  # Use FLAIR for display
seg_slice = pred_mask_3d[:, :, mid_z]

# Normalize MRI for visualization
mri_normalized = mri_slice / (mri_slice.max() + 1e-8)

# Create RGB background
overlay = np.stack([mri_normalized] * 3, axis=-1)  # Shape: (240, 240, 3)

# Highlight tumor in white
overlay[seg_slice > 0] = [1.0, 1.0, 1.0]  # White for tumor

# Plot
plt.figure(figsize=(12, 6))

# Original MRI
plt.subplot(1, 3, 1)
plt.imshow(mri_normalized, cmap='gray')
plt.contour(seg_slice, colors='red', linewidths=0.8)
plt.title(f"{PATIENT_ID} - FLAIR MRI\n(Red: Tumor Boundary)")
plt.axis('off')

# Segmentation mask
plt.subplot(1, 3, 2)
plt.imshow(seg_slice, cmap='jet', interpolation='none')
plt.title("Predicted Tumor Segmentation\n(1=Core, 2=Edema, 3=Enhancing)")
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("MRI + Tumor Overlay (White)")
plt.axis('off')

plt.tight_layout()
plt.show()

print("🖼️ Visualization displayed.")