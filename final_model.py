import pandas as pd
from scipy.sparse import hstack, csr_matrix
from konlpy.tag import Okt

# Okt 인스턴스를 모듈 최상단에서 하나만 생성
okt = Okt()

def tokenize_korean(text):
    """
    한국어 형태소 분석하여 토큰 리스트 반환.
    불용어를 제거하고 길이 2 이상인 토큰만 유지.
    """
    try:
        tokens = okt.morphs(text, stem=True)
        stopwords = [
            '은', '는', '이', '가', '을', '를', '의', '와', '과', '에',
            '에서', '으로', '로', '하다', '있다', '되다', '하는', '하고',
            '해서', '이다', '그', '것', '수', '등', '및'
        ]
        return [token for token in tokens if len(token) > 1 and token not in stopwords]
    except:
        # 형태소 분석 실패 시 공백 단위로 분리
        return text.split()

def extract_sentiment_features(text):
    positive_words = ['상승','매수','호재','급등','폭등','강세','돌파','반등','기대','성장']
    negative_words = ['하락','매도','악재','급락','폭락','약세','조정','우려','손실','위험']
    intensity_words = ['급','폭','대','초','극']
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    intensity_count = sum(1 for w in intensity_words if w in text)
    total_words = len(text.split())
    return {
        'sentiment_score': (pos_count - neg_count) / (total_words + 1),
        'sentiment_intensity': ((pos_count - neg_count)/(total_words+1)) * (intensity_count/(total_words+1)),
        'text_length': len(text),
        'word_count': total_words,
        'number_count': len([w for w in text.split() if any(c.isdigit() for c in w)]),
        'exclamation_count': text.count('!')
    }

def predict_stock_movement(text, pipeline):
    try:
        # 1) 감성 특성 추출 후 스케일링
        sentiment_result = extract_sentiment_features(text)
        sentiment_df = pd.DataFrame([sentiment_result])
        sentiment_scaled = pipeline['scaler'].transform(sentiment_df) * 3.0

        # 2) TF-IDF 벡터화
        text_vectorized = pipeline['tfidf_vectorizer'].transform([text])

        # 3) 학습 당시의 전략(best_strategy)에 따라 정확히 분기
        strategy = pipeline['best_strategy']
        
        if strategy == 'tfidf_sentiment':
            # 학습 때 TF-IDF+감성으로 했으면, 예측 때도 합쳐야 506차원
            X_new = hstack([text_vectorized, csr_matrix(sentiment_scaled)])
        
        elif strategy == 'tfidf_only':
            # 학습 때 TF-IDF만 썼다면, 500차원만 넣어야 함
            X_new = text_vectorized
        
        elif strategy == 'sentiment_only':
            # 학습 때 감성 특성만 썼다면, 감성 특성만 6차원으로 넣어야 함
            X_new = csr_matrix(sentiment_scaled)
        
        else:
            # 혹시 다른 이름(ex: 'basic', 'reduced_tfidf')으로 저장되었다면,
            # 학습 코드를 참고해 해당 이름일 때에도 TF-IDF+감성 합치는 분기를 추가하세요.
            X_new = hstack([text_vectorized, csr_matrix(sentiment_scaled)])

        # 4) 예측
        pred = pipeline['model'].predict(X_new)[0]
        proba = pipeline['model'].predict_proba(X_new)[0]
        return {
            'prediction': '📈 상승' if pred == 1 else '📉 하락',
            'prediction_class': 'up' if pred == 1 else 'down',
            'confidence': round(max(proba) * 100, 1),
            'up_probability': round(proba[1] * 100, 1),
            'down_probability': round(proba[0] * 100, 1)
        }

    except Exception as e:
        print(f"predict_stock_movement 오류: {e}")
        return None