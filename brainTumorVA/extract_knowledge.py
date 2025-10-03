# extract_knowledge.py
from pypdf import PdfReader
import os

# Path to your PDF
pdf_path = "C:/Users/Bilal/Desktop/BrainTumorVA/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

# Output text file
output_path = "data/medical_knowledge.txt"

# Create data folder
os.makedirs("data", exist_ok=True)

# Read PDF
print("📄 Extracting text from PDF...")
reader = PdfReader(pdf_path)

text = ""
for i, page in enumerate(reader.pages):
    page_text = page.extract_text()
    if page_text:  # Avoid None
        text += page_text + "\n"
    print(f"  Page {i+1}/{len(reader.pages)} processed")

# Save as UTF-8 encoded text file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Text extracted and saved to {output_path}")
print(f"📊 Total characters: {len(text)}")