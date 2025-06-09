import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 1. 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 사전학습된 감성분석 모델 및 토크나이저 불러오기 (KR-FinBert)
MODEL_NAME = "snunlp/KR-FinBert-SC"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

# 텍스트 전처리 함수
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 한글, 영문, 숫자, 공백만 남기고 제거
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    # 여러 공백 하나로
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 감성분석 함수: {'label': ..., 'score': ...} 반환
def predict_sentiment(text: str) -> dict:
    text = clean_text(text)
    if not text:
        return {"label": "중립", "score": 0.0}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    score, pred = torch.max(probs, dim=1)
    label_map = {0: "부정", 1: "중립", 2: "긍정"}
    return {"label": label_map[pred.item()], "score": round(score.item(), 4)}

