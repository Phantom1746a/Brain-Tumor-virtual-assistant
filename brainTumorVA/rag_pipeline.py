# rag_pipeline.py
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from config import GROQ_API_KEY, YOLO_REPORT_PATH, KNOWLEDGE_PATH, OUTPUT_PATH
import os

# Local FAISS directory
FAISS_INDEX = "faiss_index"

'''def generate_clinical_report():
    # --- 1. Validate Input Files ---
    if not os.path.exists(YOLO_REPORT_PATH):
        return "⚠️ No segmentation report found. Please run detection first."

    if not os.path.exists(KNOWLEDGE_PATH):
        return "⚠️ Medical knowledge base not found. Run 'extract_knowledge.py' first."

    # --- 2. Read Files Safely with UTF-8 ---
    try:
        with open(YOLO_REPORT_PATH, "r", encoding="utf-8", errors="replace") as f:
            query = f.read()
    except Exception as e:
        return f"❌ Error reading YOLO report: {str(e)}"

    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8", errors="replace") as f:
            knowledge_text = f.read()
    except Exception as e:
        return f"❌ Error reading knowledge base: {str(e)}"

    if not knowledge_text.strip():
        return "⚠️ Knowledge base is empty. Check PDF extraction."

    # --- 3. Split Text into Chunks ---
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    texts = splitter.split_text(knowledge_text)

    # --- 4. Embeddings ---
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        return f"❌ Error loading embeddings: {str(e)}"

    # --- 5. Create or Load FAISS Vector Store ---
    index_path = os.path.join(FAISS_INDEX, "index.faiss")
    try:
        if os.path.exists(index_path):
            print("📂 Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                FAISS_INDEX,
                embedding,
                allow_dangerous_deserialization=True
            )
        else:
            print("🧠 Creating new FAISS index...")
            vectorstore = FAISS.from_texts(texts, embedding)
            vectorstore.save_local(FAISS_INDEX)
            print("✅ FAISS index saved.")
    except Exception as e:
        return f"❌ Error with FAISS index: {str(e)}"

    # --- 6. Initialize LLM ---
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=512,
            timeout=30
        )
    except Exception as e:
        return f"❌ LLM initialization failed: {str(e)}"

    # --- 7. Setup Retrieval QA Chain ---
    retriever = vectorstore.as_retriever(k=2)  # Retrieve top 2 relevant chunks
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    # --- 8. Prompt for LLM ---
    prompt =  f"""
You are a medical AI assistant trained on authoritative sources including *The Gale Encyclopedia of Medicine*, a comprehensive reference work covering diseases, symptoms, diagnostics, and treatments.

Based on the tumor analysis below and general medical knowledge, generate a detailed, professional clinical report with the following structure:

---

**AI Medical Report: Brain Tumor Analysis**  


**1. Summary**  
Provide a concise overview of the imaging findings, including tumor presence, size, and location.

**2. Imaging Findings**  
Describe the radiological characteristics:
- Size ( percentage of brain area)
- Location (e.g., frontal lobe, right parietal region)
- Morphology (well-defined, irregular, etc.)
- Number of lesions

**3. Clinical Interpretation**  
Using *The Gale Encyclopedia of Medicine* as a reference:
- Discuss the most likely tumor types (e.g., glioma, meningioma, metastasis)
- Explain symptoms associated with this location
- Mention possible underlying causes or risk factors
- Differentiate between benign and malignant possibilities

**4. Recommended Next Steps**  
Suggest evidence-based next steps:
- MRI with contrast
- Neurological examination
- Referral to neurology or neurosurgery
- Biopsy or monitoring, depending on size/confidence

**5. Prognostic Notes**  
Briefly mention survival rates or outcomes based on tumor type and grade (per Gale)

Be thorough, factual, and avoid speculation. Do not hallucinate. Use professional medical language.

Tumor Analysis Input:
{query}

---
"""

    # --- 9. Get Response ---
    try:
        response = qa.run(prompt)
    except Exception as e:
        response = f"❌ Error generating response: {str(e)}\n\nEnsure your GROQ_API_KEY is valid and you have internet access."

    # --- 10. Save Response ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(response)
    except Exception as e:
        print(f"⚠️ Could not save response: {str(e)}")

    return response'''
def generate_clinical_report(patient_context=""):
    # --- 1. Validate Input Files ---
    if not os.path.exists(YOLO_REPORT_PATH):
        return "⚠️ No segmentation report found. Please run detection first."

    if not os.path.exists(KNOWLEDGE_PATH):
        return "⚠️ Medical knowledge base not found. Run 'extract_knowledge.py' first."

    # --- 2. Read Files Safely ---
    try:
        with open(YOLO_REPORT_PATH, "r", encoding="utf-8", errors="replace") as f:
            yolo_report = f.read()
    except Exception as e:
        return f"❌ Error reading YOLO report: {str(e)}"

    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8", errors="replace") as f:
            knowledge_text = f.read()
    except Exception as e:
        return f"❌ Error reading knowledge base: {str(e)}"

    if not knowledge_text.strip():
        return "⚠️ Knowledge base is empty."

    # --- 3. Split Text ---
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(knowledge_text)

    # --- 4. Embeddings ---
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        return f"❌ Error loading embeddings: {str(e)}"

    # --- 5. FAISS Index ---
    index_path = os.path.join(FAISS_INDEX, "index.faiss")
    try:
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(FAISS_INDEX, embedding, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_texts(texts, embedding)
            vectorstore.save_local(FAISS_INDEX)
    except Exception as e:
        return f"❌ Error with FAISS index: {str(e)}"

    # --- 6. LLM Setup ---
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=768,
            timeout=30
        )
    except Exception as e:
        return f"❌ LLM initialization failed: {str(e)}"

    # --- 7. RAG Chain ---
    retriever = vectorstore.as_retriever(k=2)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    # --- 8. Enhanced Prompt with Patient Context ---
    prompt = f"""
You are a medical AI assistant trained on *The Gale Encyclopedia of Medicine*. Generate a detailed clinical report using both imaging and patient data.

---

**AI Medical Report: Brain Tumor Analysis**

{patient_context}

**Imaging Findings from YOLOv11 Segmentation:**
{yolo_report}

**Instructions:**
1. Summarize the case with patient-specific insights.
2. Interpret tumor location and size in context of symptoms.
3. Discuss likely tumor types (e.g., glioma, meningioma).
4. Recommend next steps (MRI with contrast, neurology referral).
5. Mention prognosis based on age and tumor type.
6. Do not mention the version of Yolo

Be thorough, factual, and compassionate. Avoid hallucinations.
"""

    # --- 9. Get Response ---
    try:
        response = qa.run(prompt)
    except Exception as e:
        response = f"❌ Error generating response: {str(e)}\nEnsure your GROQ_API_KEY is valid."

    # --- 10. Save Response ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(response)
    except Exception as e:
        print(f"⚠️ Could not save response: {str(e)}")

    return response