# test.py

# 1) 원본 토크나이저 함수 그대로 복사
from konlpy.tag import Okt
okt = Okt()

def tokenize_korean(text):
    try:
        tokens = okt.morphs(text, stem=True)
        stopwords = ['은','는','이','가','을','를','의','와','과','에','에서','으로','로',
                     '하다','있다','되다','하는','하고','해서','이다','그','것','수','등','및']
        return [token for token in tokens if len(token) > 1 and token not in stopwords]
    except:
        return text.split()

# 2) joblib load
from joblib import load
orig_path = r"C:\Users\azureuser\Desktop\firstproject_flask\stock_prediction_model_20250610_183646.pkl"
pipeline = load(orig_path)

print("Loaded successfully!")
