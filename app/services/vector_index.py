import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

model = SentenceTransformer("all-MiniLM-L6-v2")  # Nhẹ và nhanh

INDEX_PATH = "output/index.faiss"
META_PATH = "output/enriched_metadata.json"


def build_index_from_metadata():
    with open(META_PATH) as f:
        metadata = json.load(f)

    texts = []
    new_metadata = []
    for chunk in metadata:
        if chunk["chunk_type"] == "text" and "text" in chunk:
            texts.append(chunk["text"])
            new_metadata.append(chunk)
        elif chunk["chunk_type"] == "table" and "table" in chunk:
            texts.append(chunk["table"])
            new_metadata.append(chunk)
        elif chunk["chunk_type"] == "image" and "caption" in chunk:
            caption = chunk["caption"].strip()
            if caption.lower().startswith("image on page"):
                caption = f"Illustration or diagram from page {chunk['page']} of document {chunk['doc']}"
                chunk["caption"] = caption
            texts.append(caption)
            new_metadata.append(chunk)

    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(new_metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved index with {len(texts)} entries")


def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)

    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return results


if __name__ == "__main__":
    build_index_from_metadata()
