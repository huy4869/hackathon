import json
import time
from app.services.llm_client import ask_llm
from app.services.index_builder import search_top_chunks

MAX_CHUNK_LEN = 1000

def truncate(text, max_len=MAX_CHUNK_LEN):
    return text if len(text) <= max_len else text[:max_len] + "..."

def build_prompt(question, chunks):
    context_parts = []
    for chunk in chunks:
        if chunk["chunk_type"] == "text":
            context_parts.append(f"[Text - {chunk['doc']}, page {chunk['page']}]\n{truncate(chunk['text'])}")
        elif chunk["chunk_type"] == "table":
            context_parts.append(f"[Table - {chunk['doc']}, page {chunk['page']}]\n{truncate(chunk['table'])}")
        elif chunk["chunk_type"] == "image":
            context_parts.append(f"[Image - {chunk['doc']}, page {chunk['page']}]\nCaption: {chunk['caption']}")

    context_str = "\n\n".join(context_parts)

    prompt = f"""
You are an assistant helping answer questions based on official process documentation.

You will be given a question and relevant snippets (text, tables, image captions) extracted from PDFs.

Your task is to:
- Read the context carefully.
- Answer the question truthfully and clearly.
- Only use information from the provided context. Do not invent facts.
- Format the output as structured JSON with keys: answer_text, citations, images, tables.

Each citation should have the doc name and page number.
If no answer can be found in the context, respond exactly with the following JSON object:
{{
  "answer_text": "Tôi không tìm thấy thông tin phù hợp trong tài liệu.",
  "citations": [],
  "images": [],
  "tables": []
}}

Question:
{question}

Context:
{context_str}

Return only the following JSON object, with no markdown formatting, no explanation, and no extra characters:

{{
  "answer_text": "...",
  "citations": [ {{ "doc": "...", "page": ... }}, ... ],
  "images": [ {{ "caption": "...", "b64_png": "..." }}, ... ],
  "tables": [ {{ "caption": "...", "markdown": "..." }}, ... ]
}}
""".strip()

    return prompt

def build_answer(question, top_k=3):
    if len(question.strip().split()) < 3:
        print("⚠️ Warning: câu hỏi quá ngắn, có thể LLM không hiểu ý định.")

    start_time = time.time()

    top_chunks = search_top_chunks(question, top_k=top_k)
    search_done = time.time()
    print(f"🔍 Search done in {search_done - start_time:.2f}s")

    prompt = build_prompt(question, top_chunks)
    prompt_done = time.time()
    print(f"📄 Prompt built in {prompt_done - search_done:.2f}s")

    context_str = prompt.split("Context:", 1)[-1].strip()
    raw_answer = ask_llm(context_str, question)
    llm_done = time.time()
    print(f"🤖 LLM answered in {llm_done - prompt_done:.2f}s")

    total_time = llm_done - start_time
    print(f"⏱ Tổng thời gian xử lý: {total_time:.2f}s")

    try:
        answer_start = raw_answer.find('{')
        answer_end = raw_answer.rfind('}') + 1
        answer_json_str = raw_answer[answer_start:answer_end]
        answer_obj = json.loads(answer_json_str)

        existing_captions = set(img.get("caption", "").strip() for img in answer_obj.get("images", []))

        for chunk in top_chunks:
            if chunk["chunk_type"] == "image":
                actual_caption = chunk.get("caption", f"image on page {chunk['page']}").strip()
                if actual_caption not in existing_captions:
                    answer_obj["images"].append({
                        "caption": actual_caption,
                        "b64_png": chunk.get("image_b64", "")
                    })
                    existing_captions.add(actual_caption)

        enrich_done = time.time()
        print(f"🖼️ Enrich images done in {enrich_done - llm_done:.2f}s")
        print(f"⏱ Tổng thời gian hoàn tất tất cả bước: {enrich_done - start_time:.2f}s")

        return answer_obj

    except Exception as e:
        error_time = time.time()
        print(f"❌ Error parsing answer in {error_time - llm_done:.2f}s")
        return {
            "answer_text": "Tôi không tìm thấy thông tin phù hợp trong tài liệu.",
            "citations": [],
            "images": [],
            "tables": [],
            "debug": raw_answer,
            "error": str(e)
        }
