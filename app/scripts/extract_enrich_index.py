import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
import faiss
import pytesseract
import numpy as np
import pdfplumber

PDF_DIR = "data"
OUT_DIR = "output"
PAGES_DIR = os.path.join(OUT_DIR, "pages")
META_OUT = os.path.join(OUT_DIR, "enriched_metadata.json")
INDEX_OUT = os.path.join(OUT_DIR, "index.faiss")

os.makedirs(PAGES_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

metadata = []
embeddings = []

for filename in os.listdir(PDF_DIR):
    if not filename.endswith(".pdf"):
        continue

    doc_path = os.path.join(PDF_DIR, filename)
    doc_name = os.path.splitext(filename)[0]
    pdf = fitz.open(doc_path)
    pdf_plumber = pdfplumber.open(doc_path)

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        page_plumber = pdf_plumber.pages[page_num]
        text = page.get_text()

        try:
            label = int(page.get_label())
        except:
            label = page_num + 2  # Adjust for page label shift

        # Extract images (base64 only)
        images = page.get_images(full=True)
        image_chunks = []
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                pix = fitz.Pixmap(pdf, xref)
                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                image_bytes = pix.tobytes("png")

                img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                if img_pil.width * img_pil.height < 1000:
                    continue

                buffered = BytesIO()
                img_pil.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                try:
                    ocr_text = pytesseract.image_to_string(img_pil).strip()
                except:
                    ocr_text = ""

                if len(ocr_text.strip()) < 1:
                    if "workflow" in text.lower():
                        ocr_text = f"Diagram about workflow extracted from page {label} of document '{doc_name}'"
                    else:
                        ocr_text = f"An illustration, diagram, or image extracted from page {label} of document '{doc_name}'"
                else:
                    ocr_text = f"Image from page {label} of document '{doc_name}' — OCR extracted: {ocr_text.strip()}"

                image_chunks.append({"b64": img_b64, "caption": ocr_text})
            except Exception as e:
                print(f"❌ Failed to render image xref={xref} page={label}: {e}")

        # Extract tables
        tables = []
        try:
            table_data = page_plumber.extract_tables()
            for table in table_data:
                if not table or len(table) < 2:
                    continue
                df = pd.DataFrame(table[1:], columns=table[0])
                markdown = df.to_markdown(index=False)
                tables.append({"caption": f"Table on page {label}", "markdown": markdown})
        except Exception as e:
            print(f"⚠️ Error extracting table on page {label}: {e}")

        # Save per-page info
        out_json = {
            "text": text.strip(),
            "tables": tables,
            "images": image_chunks  # changed to keep full image object for caption and b64
        }
        page_filename = f"{doc_name}_page_{label}.json"
        with open(os.path.join(PAGES_DIR, page_filename), "w") as f:
            json.dump(out_json, f, indent=2, ensure_ascii=False)

        # Build metadata
        if len(text.strip()) > 10:
            metadata.append({"doc": doc_name, "page": label, "chunk_type": "text", "text": text.strip()})
            embeddings.append(model.encode(text.strip()))
        for tbl in tables:
            metadata.append({"doc": doc_name, "page": label, "chunk_type": "table", "table": tbl["markdown"]})
            embeddings.append(model.encode(tbl["markdown"]))
        for img in image_chunks:
            metadata.append({"doc": doc_name, "page": label, "chunk_type": "image", "image_b64": img["b64"], "caption": img["caption"]})
            embeddings.append(model.encode(img["caption"]))

        # Holistic page-level embedding
        page_summary = text.strip() + "\n\n" + "\n\n".join([tbl["markdown"] for tbl in tables]) + "\n\n" + "\n\n".join([img["caption"] for img in image_chunks])
        metadata.append({"doc": doc_name, "page": label, "chunk_type": "page_summary", "summary": page_summary})
        embeddings.append(model.encode(page_summary))

    pdf.close()
    pdf_plumber.close()

with open(META_OUT, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, INDEX_OUT)

print("✅ Extracted, enriched and indexed successfully")
