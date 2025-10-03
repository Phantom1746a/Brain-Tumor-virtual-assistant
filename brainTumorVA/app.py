# app.py
import streamlit as st
import tempfile
import os
import time
from yolo_segmentation import run_segmentation
from rag_pipeline import generate_clinical_report
from pdf_generator import create_pdf  # Uses reportlab

# Page config
st.set_page_config(
    page_title="🧠 Brain Tumor Analyzer",
    page_icon="🧠",
    layout="centered"
)

# Title & Description
st.title("🧠 AI Brain Tumor Analysis")
st.markdown("""
Upload a brain MRI and provide patient details for AI-powered tumor detection and clinical interpretation.
""")

# Sidebar
st.sidebar.header("📤 Upload MRI")
uploaded_file = st.sidebar.file_uploader("Choose an MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("🖼️ Uploaded MRI")
    st.image(uploaded_file, caption="Original MRI", use_container_width=True)

    # --- PATIENT & MEDICAL FORM ---
    st.subheader("🧑 Patient Information")
    with st.form("patient_form"):
        name = st.text_input("Patient Name", placeholder="e.g., John Doe")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        st.write("### 🩺 Clinical Symptoms (Check all that apply)")
        symptoms = []
        if st.checkbox("Headache (especially in the morning)"): symptoms.append("Headache")
        if st.checkbox("Nausea or vomiting"): symptoms.append("Nausea/Vomiting")
        if st.checkbox("Seizures"): symptoms.append("Seizures")
        if st.checkbox("Vision changes"): symptoms.append("Vision Changes")
        if st.checkbox("Weakness or numbness in limbs"): symptoms.append("Motor Weakness")
        if st.checkbox("Speech difficulties"): symptoms.append("Speech Problems")
        if st.checkbox("Memory or cognitive changes"): symptoms.append("Cognitive Decline")
        if st.checkbox("Balance or coordination problems"): symptoms.append("Ataxia")

        st.write("### 📚 Medical History")
        has_head_injury = st.checkbox("Recent head injury?")
        prior_cancer = st.text_input("History of cancer?", placeholder="e.g., Lung cancer (2022)")
        neurological_disorder = st.text_input("Neurological disorder?", placeholder="e.g., Epilepsy")

        submitted = st.form_submit_button("🔍 Analyze Tumor")

    if submitted:
        if not name.strip():
            st.error("❌ Please enter the patient's name.")
        else:
            # Save uploaded image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_path = tmp_file.name

            try:
                # --- Step 1: YOLO Segmentation ---
                with st.spinner("🔍 Detecting tumor with YOLOv11-Segmentation Model..."):
                    segmented_img_path = run_segmentation(input_path)

                    if segmented_img_path is None:
                        st.error("❌ Could not process image. Is it a valid MRI?")
                        st.stop()

                    if segmented_img_path == "no_tumor":
                        st.info("🟢 No tumor detected by the AI model.")
                        st.stop()

                st.success("✅ Tumor segmented successfully!")

                # --- Step 2: Show Segmented Image ---
                st.subheader("🎨 Tumor Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(input_path, caption="Original MRI", use_container_width=True)
                with col2:
                    st.image(segmented_img_path, caption="Tumor Highlighted", use_container_width=True)

                # --- Step 3: Prepare Patient Context ---
                patient_context = f"""
**Patient Information:**
- Name: {name}
- Age: {age}
- Gender: {gender}

Reported Symptoms:
{', '.join(symptoms) if symptoms else 'None reported'}

Medical History:
- Head Injury: {'Yes' if has_head_injury else 'No'}
- Prior Cancer: {prior_cancer or 'None'}
- Neurological Disorder: {neurological_disorder or 'None'}
                """.strip()

                # --- Step 4: Generate LLM Clinical Insight ---
                with st.spinner("🧠 Generating personalized clinical interpretation..."):
                    llm_response = generate_clinical_report(patient_context=patient_context)

                # --- Step 5: Display LLM Response ---
                st.subheader("📄 AI Clinical Report")
                st.write(llm_response)

                # --- Step 6: Generate PDF Report ---
                with st.spinner("📄 Generating PDF report..."):
                    pdf_path = create_pdf(
                        original_img_path=input_path,
                        segmented_img_path=segmented_img_path,
                        llm_response=llm_response,
                        patient_data=patient_context
                    )

                if pdf_path and os.path.exists(pdf_path):
                    st.success("✅ PDF report generated!")
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=f"report_{name.replace(' ', '_')}_{int(time.time())}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.warning("⚠️ Could not generate PDF. Check logs for details.")

            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.exception(e)

            finally:
                # Cleanup temp files
                try:
                    os.unlink(input_path)
                    if 'segmented_img_path' in locals() and os.path.exists(segmented_img_path):
                        os.unlink(segmented_img_path)
                except Exception as cleanup_error:
                    st.warning(f"⚠️ Could not clean up temp files: {cleanup_error}")