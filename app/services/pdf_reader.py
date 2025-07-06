import fitz  # PyMuPDF
import pdfplumber
import os
import base64
import pandas as pd
from pathlib import Path

def extract_text_and_tables(pdf_path: str, output_dir: str):
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem
    results = []

    with pdfplumber.open(pdf_path) as plumber_doc:
        for i, page in enumerate(plumber_doc.pages):
            page_num = i + 1
            text = page.extract_text()
            tables = page.extract_tables()

            table_markdowns = []
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                markdown = df.to_markdown(index=False)
                table_markdowns.append(markdown)

            results.append({
                "doc": pdf_name,
                "page": page_num,
                "text": text,
                "tables": table_markdowns,
            })

    return results
