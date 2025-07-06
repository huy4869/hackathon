import os
import faiss
import pickle
import json
import numpy as np

from sentence_transformers import SentenceTransformer

# Đường dẫn FAISS index và metadata
INDEX_PATH = "output/index.faiss"
META_PATH = "output/enriched_metadata.json"

# Khởi tạo model embedding
model = SentenceTransformer("all-MiniLM-L6-v2")  # Nhẹ, nhanh, độ chính xác tốt

# Biến toàn cục
faiss_index = None
metadata = None

def load_index():
    global faiss_index, metadata

    if faiss_index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"Không tìm thấy FAISS index tại {INDEX_PATH}")
        faiss_index = faiss.read_index(INDEX_PATH)

    if metadata is None:
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Không tìm thấy metadata tại {META_PATH}")
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

def get_embedding(text: str) -> np.ndarray:
    """
    Encode text thành vector dùng sentence-transformers
    """
    emb = model.encode([text], normalize_embeddings=True)
    return emb[0].astype("float32")

def search_top_chunks(question: str, top_k: int = 3):
    """
    Trả về top_k đoạn văn bản gần nhất với câu hỏi
    """
    load_index()

    query_emb = get_embedding(question)
    D, I = faiss_index.search(np.array([query_emb]), top_k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])

    return results
