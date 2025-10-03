# config.py
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

Roboflow_key=os.getenv("Roboflow_key")
# 🔑 1. Validate Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is missing in .env file! Please add it.")

# 📁 2. Load required paths from .env
YOLO_REPORT_PATH = os.getenv("YOLO_REPORT_PATH")
if not YOLO_REPORT_PATH:
    raise ValueError("❌ YOLO_REPORT_PATH is not set in .env")

KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH")
if not KNOWLEDGE_PATH:
    raise ValueError("❌ KNOWLEDGE_PATH is not set in .env")

OUTPUT_PATH = os.getenv("OUTPUT_PATH")
if not OUTPUT_PATH:
    raise ValueError("❌ OUTPUT_PATH is not set in .env")

PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR")
if not PDF_OUTPUT_DIR:
    raise ValueError("❌ PDF_OUTPUT_DIR is not set in .env")

# 📂 3. Fixed FAISS index directory (do not get from .env)
FAISS_INDEX_DIR = "faiss_index"  # Local folder — no need for .env

# 🛠️ 4. Create all required directories
directories_to_create = [
    os.path.dirname(YOLO_REPORT_PATH),   # e.g., reports/
    os.path.dirname(OUTPUT_PATH),        # e.g., outputs/
    PDF_OUTPUT_DIR,                      # e.g., outputs/pdf_reports/
    FAISS_INDEX_DIR                      # e.g., faiss_index/
]

for path in directories_to_create:
    if path and path.strip():  # Avoid empty or whitespace-only paths
        os.makedirs(path, exist_ok=True)
        # Optional: Uncomment for debug
        # print(f"📁 Created or verified directory: {path}")

# ✅ Expose only what's needed
__all__ = [
    "GROQ_API_KEY",
    "YOLO_REPORT_PATH",
    "KNOWLEDGE_PATH",
    "OUTPUT_PATH",
    "PDF_OUTPUT_DIR",
    "FAISS_INDEX_DIR"
]