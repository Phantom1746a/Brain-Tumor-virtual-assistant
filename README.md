# 🧠 Brain Tumor Analysis System

An AI-powered pipeline for brain tumor detection, segmentation, and clinical report generation.

## 🔍 Overview

1. **Tumor Detection with YOLO**  
   - Uses YOLOv11-Segmentation to detect tumor regions in MRI scans.
   - Fast 2D slice-level detection for initial localization.

2. **Precise Segmentation with 3D U-Net**  
   - A 3D Attention U-Net model trained on the **BraTS 2020 dataset** refines segmentation.
   - Identifies sub-regions: Necrotic/Core, Edema, and Enhancing Tumor.

3. **Clinical Report Generation (RAG + LLM)**  
   - Segmentation results are sent to a **Retrieval-Augmented Generation (RAG)** pipeline.
   - The system uses a **medical knowledge base** (from *The Gale Encyclopedia of Medicine*).
   - Powered by Groq's LLMs: `llama-3.3-70b-versatile` or `meta-llama/llama-4-scout-17b-16e-instruct`.
   - Generates clinically relevant insights and recommendations.

4. **Patient Input at Runtime**  
   - Users provide:
     - Patient name
     - Age, gender
     - Symptoms (headache, seizures, etc.)
     - Medical history (prior cancer, head injury, neurological disorders)
   - This information is included in the final report for personalized analysis.

5. **Final Output: Professional PDF Report**  
   - Combines:
     - Patient details
     - MRI & segmentation images
     - AI-generated clinical interpretation
   - Exported using `reportlab`
   - Ready for review or archiving

## ⚠️ Model Updates (Groq Deprecations)
As per [Groq deprecations](https://console.groq.com/docs/deprecations):
- `mixtral-8x7b-32768` → use `llama-3.3-70b-versatile` or `mistral-saba-24b`
- `llava-v1.5-7b` → use `llama-3.2-11b-vision-preview`
- New recommendation: `meta-llama/llama-4-scout-17b-16e-instruct`

## 📚 Data Source
- **MRI Data**: [BraTS 2020 Challenge](https://www.med.upenn.edu/cbica/brats2020/)
- Focus: Glioma segmentation in multimodal MRI (FLAIR, T1, T1ce, T2)


streamlit run app.py


---

✅ Clean, focused, and explains exactly how your system works — from **YOLO detection** → **U-Net segmentation** → **RAG + medical history** → **PDF report**.
