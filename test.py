

from app.services.answer_builder import build_answer
import json

result = build_answer("Who approves emergency changes?")
print(json.dumps(result, indent=2, ensure_ascii=False))