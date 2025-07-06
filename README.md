source venv/bin/activate
run app: uvicorn main:app --reload
run faiss: python3 app/scripts/extract_enrich_index.py

swagger: http://127.0.0.1:8000/docs

------------------------------------------------------------------------------------------------------------------------------------------
Tham số "top_k": 3 có ý nghĩa là:

🔍 Trích xuất 3 đoạn thông tin (snippet) liên quan nhất từ tài liệu để đưa vào context cho mô hình LLM tạo câu trả lời.

Cụ thể hơn:

Khi bạn hỏi 1 câu hỏi (question), hệ thống dùng FAISS để tìm các đoạn văn bản (text), bảng (table), hoặc ảnh (image caption) có embedding gần nhất với câu hỏi đó.

top_k = 3 nghĩa là: chỉ lấy 3 đoạn (chunk) có điểm tương đồng cao nhất để đưa vào prompt → mô hình chỉ đọc 3 đoạn này để trả lời.

🎯 Mục tiêu:

Giúp LLM tập trung vào nội dung liên quan → tránh quá dài, lạc đề.

Đảm bảo tốc độ trả lời nhanh (dưới 10 giây như đề bài yêu cầu).

Tránh vượt quá context_length của mô hình (ví dụ Gemma 3n hỗ trợ 32k tokens nhưng tốt nhất nên giữ ngắn gọn).

📌 Bạn có thể thay đổi:

top_k = 1 nếu muốn cực nhanh và súc tích.

top_k = 5 nếu muốn tăng độ chính xác và đầy đủ hơn (nếu tài liệu dài hoặc LLM mạnh hơn).

------------------------------------------------------------------------------------------------------------------------------------------

Question có table
{
  "question": "What is the process workflow of Service Configuration Management?",
  "top_k": 2
}

Question có image

{
  "question": "Show me diagrams related to risk management workflow.",
  "top_k": 3
}
