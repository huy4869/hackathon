from llama_cpp import Llama
import os

# Chỉ load 1 lần
from pathlib import Path

# Đường dẫn tuyệt đối đến mô hình, bất kể file đang ở đâu
LLM_PATH = str(Path(__file__).resolve().parents[2] / "models" / "gemma-3n-e4b-it.gguf")

print("👉 Loading model from:", LLM_PATH)
assert Path(LLM_PATH).exists(), f"❌ Model file not found: {LLM_PATH}"


# LLM_PATH = "models/gemma-3n-e4b-it.gguf"


llm = Llama(
    # model_path=LLM_PATH,    
    # n_ctx=2048,
    # n_threads=6,
    # n_gpu_layers=20,  # tùy máy, bạn có thể giảm nếu RAM thấp
    # verbose=False
    model_path=LLM_PATH,
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=20,
    use_mmap=True,
    use_mlock=False,
    batch_size=1024,   # hoặc 512 tùy RAM
    verbose=False
)

def build_prompt(context: str, question: str) -> str:
    prompt = f"""Bạn là trợ lý AI thông minh. Dựa trên ngữ cảnh bên dưới, hãy trả lời câu hỏi một cách ngắn gọn và tự nhiên như con người. Nếu không có thông tin, hãy trả lời "Tôi không tìm thấy thông tin phù hợp trong tài liệu".

Ngữ cảnh:
{context}

Câu hỏi: {question}
Trả lời:"""
    return prompt

def ask_llm(context: str, question: str) -> str:
    prompt = build_prompt(context, question)
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        top_p=0.95,
        stop=["Câu hỏi:", "Ngữ cảnh:"]
    )
    return response["choices"][0]["text"].strip()


