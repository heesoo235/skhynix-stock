# resave_model.py

from joblib import load, dump
import final_model            # 최종적으로 토크나이저가 이 모듈에 있어야 함
from final_model import tokenize_korean

# 1) 원본 모델(pkl) 경로
orig_path = r"C:\Users\azureuser\Desktop\firstproject_flask\stock_prediction_model_20250610_183646.pkl"

# 2) 모델 로드 (이 시점에도 __main__.tokenize_korean 참조 중이어서 잠깐만 로드 가능)
pipeline = load(orig_path)

# ────────────────────────────────────────────────────────
# 3) TF-IDF 벡터라이저의 tokenizer를 반드시 final_model.tokenize_korean으로 덮어씁니다.
#    (pickle 내부에 저장된 함수 참조를 실제 모듈 경로로 바꾸는 핵심 단계)
tfidf = pipeline['tfidf_vectorizer']
# 벡터라이저의 tokenizer 속성이 존재하면 덮어쓰기
if hasattr(tfidf, 'tokenizer'):
    tfidf.tokenizer = tokenize_korean
else:
    # 만약 TfidfVectorizer 사용 시 tokenizer가 아닌 analyzer 형태였다면,
    # 예: tfidf_analyzer = tfidf.build_analyzer() 같은 코드가 필요할 수도 있음.
    pass

# 4) 수정된 pipeline을 새로운 파일로 저장
fixed_path = r"C:\Users\azureuser\Desktop\firstproject_flask\stock_model_fixed.pkl"
dump(pipeline, fixed_path)

print(f"✅ 모델 재저장 완료: {fixed_path}")
