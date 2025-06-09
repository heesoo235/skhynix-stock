import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime, timedelta

# 데이터 로드
board_df = pd.read_csv('/Workspace/Users/1dt017@msacademy.msai.kr/labeled_stock_data.csv')

def preprocess_and_summarize(board_df):
    clean_df = board_df[
        (board_df['body'].notna()) & 
        (board_df['direction_label'].notna())
    ].copy()
    
    clean_df['date'] = pd.to_datetime(clean_df['date'])
    clean_df['content'] = clean_df['title'].fillna('') + ' ' + clean_df['body'].fillna('')
    
    daily_data = clean_df.groupby(clean_df['date'].dt.date).agg({
        'content': lambda x: ' '.join(x.astype(str)[:30]),
        'direction_label': 'first'
    }).reset_index()
    
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data = daily_data.sort_values('date').reset_index(drop=True)
    
    next_day_labels = []
    for i in range(len(daily_data)):
        current_date = daily_data.iloc[i]['date']
        next_day = get_next_trading_day(current_date)
        next_data = daily_data[daily_data['date'] == next_day]
        next_day_labels.append(next_data.iloc[0]['direction_label'] if len(next_data) > 0 else None)
    
    daily_data['next_day_label'] = next_day_labels
    final_data = daily_data[daily_data['next_day_label'].notna()].copy()
    
    print(f"전체 일별 데이터: {len(daily_data)}일")
    print(f"예측 가능한 데이터: {len(final_data)}일")
    print(f"상승 예측: {final_data['next_day_label'].sum()}개")
    print(f"하락 예측: {len(final_data) - final_data['next_day_label'].sum()}개")
    
    return final_data

def get_next_trading_day(current_date):
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day

def setup_summarization():
    tokenizer_sum = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
    model_sum = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
    return tokenizer_sum, model_sum

def summarize_text(text, tokenizer_sum, model_sum, max_length=128):
    if len(text.strip()) < 10:
        return text
    try:
        inputs = tokenizer_sum.encode(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model_sum.generate(
            inputs, max_length=max_length, min_length=10,
            length_penalty=2.0, num_beams=4, early_stopping=True
        )
        summary = tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)
        return summary if summary.strip() else text[:200]
    except:
        return text[:200]

def create_summaries(final_data, tokenizer_sum, model_sum):
    summaries = []
    for content in final_data['content']:
        summary = summarize_text(content, tokenizer_sum, model_sum)
        summaries.append(summary)
    return summaries

def setup_kobert():
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertModel.from_pretrained('monologg/kobert')
    model.eval()
    return tokenizer, model

def create_embeddings(summaries, tokenizer, model):
    embeddings = []
    for summary in summaries:
        inputs = tokenizer(summary, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(embedding)
    return embeddings

class TimeSeriesDataset(Dataset):
    def __init__(self, embeddings, labels, sequence_length=5):
        self.embeddings = embeddings
        self.labels = labels
        self.seq_len = sequence_length
    
    def __len__(self):
        return len(self.embeddings) - self.seq_len
    
    def __getitem__(self, idx):
        X = np.array([self.embeddings[i] for i in range(idx, idx + self.seq_len)])
        y = self.labels[idx + self.seq_len]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)

def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_acc = 0
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            y = y.unsqueeze(1)
            output = model(X)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_loss, _, val_acc = evaluate_model(model, val_loader, return_loss=True)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return model

def evaluate_model(model, dataloader, return_loss=False):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for X, y in dataloader:
            y = y.unsqueeze(1)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            y_true.extend(y.numpy())
            y_pred.extend((output.squeeze().numpy() > 0.5).astype(int))

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    if return_loss:
        return total_loss / len(dataloader), f1, acc
    return f1, acc

# 예측용 함수 추가
def preprocess_input_texts(input_texts, tokenizer_sum, model_sum, tokenizer_emb, model_emb):
    summaries = [summarize_text(text, tokenizer_sum, model_sum) for text in input_texts]
    embeddings = []

    for summary in summaries:
        inputs = tokenizer_emb(summary, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model_emb(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(embedding)
    
    return embeddings

def predict_direction(input_texts, trained_model, tokenizer_sum, model_sum, tokenizer_emb, model_emb, sequence_length=5):
    assert len(input_texts) >= sequence_length
    recent_texts = input_texts[-sequence_length:]
    embeddings = preprocess_input_texts(recent_texts, tokenizer_sum, model_sum, tokenizer_emb, model_emb)
    input_tensor = torch.tensor([embeddings], dtype=torch.float32)

    trained_model.eval()
    with torch.no_grad():
        output = trained_model(input_tensor)
        prob = output.item()
        prediction = 1 if prob >= 0.5 else 0

    return {'probability': prob, 'prediction': prediction, 'label': '상승' if prediction == 1 else '하락'}

# 메인 실행
def main():
    print("=== 주가 예측 시작 ===")
    final_data = preprocess_and_summarize(board_df)
    if len(final_data) < 20:
        print("데이터 부족")
        return

    tokenizer_sum, model_sum = setup_summarization()
    summaries = create_summaries(final_data, tokenizer_sum, model_sum)
    
    tokenizer_emb, model_emb = setup_kobert()
    embeddings = create_embeddings(summaries, tokenizer_emb, model_emb)

    sequence_length = 5
    dataset = TimeSeriesDataset(embeddings, final_data['next_day_label'].tolist(), sequence_length)
    if len(dataset) < 50:
        print("시계열 데이터 부족")
        return
    
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = BiLSTMClassifier()
    model = train_model(model, train_loader, val_loader)
    
    f1, acc = evaluate_model(model, val_loader)
    print(f"F1 Score: {f1:.4f} | Accuracy: {acc:.4f}")

    baseline_acc = max(final_data['next_day_label'].mean(), 1 - final_data['next_day_label'].mean())
    print(f"Baseline Accuracy: {baseline_acc:.4f} | Improvement: {acc - baseline_acc:.4f}")

    return model, tokenizer_sum, model_sum, tokenizer_emb, model_emb

# 실행
if __name__ == "__main__":
    trained_model, tokenizer_sum, model_sum, tokenizer_emb, model_emb = main()
