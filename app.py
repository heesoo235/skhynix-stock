from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import psycopg2
from psycopg2.extras import DictCursor
import os
import torch
from datetime import datetime
from dotenv import load_dotenv
from markupsafe import Markup
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from finbert import predict_sentiment 
from flask import render_template, request, redirect, url_for, session
from datetime import date
import pandas as pd
import psycopg2
from model import BiLSTMClassifier, setup_summarization, setup_kobert, predict_direction, summarize_text
from collections import Counter
import pytz
from scipy.sparse import hstack, csr_matrix
from joblib import load, dump
from final_model import predict_stock_movement


load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

def tokenize_korean(text):
    from konlpy.tag import Okt
    okt = Okt()
    return okt.morphs(text)

# model_pipeline = load(r"C:\Users\azureuser\Desktop\firstproject_flask\stock_prediction_model_20250610_183646.pkl")
MODEL_PATH = r"C:\Users\azureuser\Desktop\firstproject_flask\stock_model_fixed.pkl"
model_pipeline = load(MODEL_PATH) 

model = model_pipeline['model']
tfidf = model_pipeline['tfidf_vectorizer']
scaler = model_pipeline['scaler']
strategy = model_pipeline['best_strategy']

# app.secret_key = os.urandom(24)

def load_name_to_ticker_map(csv_path='kor_stock_list.csv'):
    df = pd.read_csv(csv_path)
    name2ticker = dict(zip(df['name'], df['yf_ticker']))
    ticker2name = dict(zip(df['yf_ticker'], df['name']))
    return name2ticker, ticker2name

NAME_TO_TICKER, TICKER_TO_NAME = load_name_to_ticker_map('kor_stock_list.csv')

def parse_input_to_ticker_list(user_input):
    tickers = []
    for n in [s.strip() for s in user_input.split(',')]:
        if n in NAME_TO_TICKER:
            tickers.append(NAME_TO_TICKER[n])
        else:
            tickers.append(n)
    return tickers

@app.route('/api/stock_multi', methods=['GET'])
def get_stock_multi():
    user_input = request.args.get('tickers', '삼성전자,LG에너지솔루션')
    days = request.args.get('days', '5')
    ticker_list = parse_input_to_ticker_list(user_input)
    result = []
    for ticker in ticker_list:
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period=f"{days}d")
            if hist.empty:
                continue
            recs = hist[['Open','Close','High','Low','Volume']].reset_index().to_dict(orient='records')
            result.append({
                "stock_name": TICKER_TO_NAME.get(ticker, ticker),
                "ticker": ticker,
                "data": recs
            })
        except Exception as e:
            continue
    return jsonify(result)


@app.template_filter('nl2br')
def nl2br(s):
    if not s:
        return ""
    return Markup(s.replace('\n', '<br>'))


def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        sslmode='require'
    )
    conn.autocommit = True
    return conn

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/create/', methods=['GET'])
def create_form():
    return render_template('create.html')

@app.route('/create/', methods=['POST'])
def create_post():
    title = request.form.get('title')
    author = request.form.get('author')
    content = request.form.get('content')

    if not title or not author or not content:
        flash('모든 필드를 채워주세요!')
        return redirect(url_for('create_form'))

    # 1. 서울 시간으로 현재시각 생성
    seoul_tz = pytz.timezone('Asia/Seoul')
    now_utc = datetime.now(pytz.UTC)

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    # 2. created_at 컬럼에 서울시간 저장
    cursor.execute(
        "INSERT INTO board.posts (title, content, author, created_at) VALUES (%s, %s, %s, %s) RETURNING id",
        (title, content, author, now_utc)
    )
    post_id = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    flash('게시글이 등록되었습니다.')
    return redirect(url_for('view_post', post_id=post_id))

@app.route('/post/<int:post_id>')
def view_post(post_id):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)

    # 1. 조회수 증가
    cursor.execute('UPDATE board.posts SET view_count = view_count + 1 WHERE id = %s', (post_id,))

    # 2. 게시글 데이터(KST 변환)
    cursor.execute("""
        SELECT 
            id, title, content, author, 
            created_at AT TIME ZONE 'Asia/Seoul' AS created_at, 
            updated_at AT TIME ZONE 'Asia/Seoul' AS updated_at,
            view_count, like_count
        FROM board.posts
        WHERE id = %s
    """, (post_id,))
    post = cursor.fetchone()
    if post is None:
        cursor.close()
        conn.close()
        flash('게시글이 없습니다.')
        return redirect(url_for('index'))

    # 3. 댓글 리스트(KST 변환)
    cursor.execute("""
        SELECT 
            id, post_id, author, content, 
            created_at AT TIME ZONE 'Asia/Seoul' AS created_at
        FROM board.comments
        WHERE post_id = %s
        ORDER BY created_at
    """, (post_id,))
    comments = cursor.fetchall()

    cursor.close()
    conn.close()

    # 4. 좋아요 체크
    user_ip = request.remote_addr
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM board.likes WHERE post_id = %s AND user_ip = %s', (post_id, user_ip))
    liked = cursor.fetchone()[0] > 0
    cursor.close()
    conn.close()

    return render_template('view.html', post=post, comments=comments, liked=liked)

@app.route('/edit/<int:post_id>', methods=['GET'])
def edit_form(post_id):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute('SELECT * FROM board.posts WHERE id = %s', (post_id,))
    post = cursor.fetchone()
    cursor.close()
    conn.close()
    if post is None:
        flash('게시글이 없습니다.')
        return redirect(url_for('index'))
    return render_template('edit.html', post=post)

@app.route('/edit/<int:post_id>', methods=['POST'])
def edit_post(post_id):
    title = request.form.get('title')
    content = request.form.get('content')
    if not title or not content:
        flash('제목과 내용을 모두 입력하세요.')
        return redirect(url_for('edit_form', post_id=post_id))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE board.posts SET title = %s, content = %s, updated_at = %s WHERE id = %s',
        (title, content, datetime.now(), post_id)
    )
    cursor.close()
    conn.close()
    flash('게시글이 수정되었습니다.')
    return redirect(url_for('view_post', post_id=post_id))

@app.route('/delete/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM board.posts WHERE id = %s', (post_id,))
    cursor.close()
    conn.close()
    flash('게시글이 삭제되었습니다.')
    return redirect(url_for('index'))

@app.route('/post/comment/<int:post_id>', methods=['POST'])
def add_comment(post_id):
    author = request.form.get('author')
    content = request.form.get('content')
    if not author or not content:
        flash('작성자와 내용을 모두 입력하세요.')
        return redirect(url_for('view_post', post_id=post_id))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO board.comments (post_id, author, content) VALUES (%s, %s, %s)',
        (post_id, author, content)
    )
    cursor.close()
    conn.close()
    flash('댓글이 등록되었습니다.')
    return redirect(url_for('view_post', post_id=post_id))

@app.route('/post/like/<int:post_id>', methods=['POST'])
def like_post(post_id):
    user_ip = request.remote_addr
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM board.likes WHERE post_id = %s AND user_ip = %s', (post_id, user_ip))
    already_liked = cursor.fetchone()[0] > 0
    if already_liked:
        cursor.execute('DELETE FROM board.likes WHERE post_id = %s AND user_ip = %s', (post_id, user_ip))
        cursor.execute('UPDATE board.posts SET like_count = like_count - 1 WHERE id = %s', (post_id,))
        message = '좋아요가 취소되었습니다.'
    else:
        cursor.execute('INSERT INTO board.likes (post_id, user_ip) VALUES (%s, %s)', (post_id, user_ip))
        cursor.execute('UPDATE board.posts SET like_count = like_count + 1 WHERE id = %s', (post_id,))
        message = '좋아요가 등록되었습니다.'
    cursor.close()
    conn.close()
    flash(message)
    return redirect(url_for('view_post', post_id=post_id))


# @app.route('/stock_predict', methods=['GET', 'POST'])
# def stock_predict():
#     prediction = None
#     sentiment_label = None
#     sentiment_score = None
#     post_sentiments = []
#     current_price = None
#     current_volume = None
#     chart_data = None

#     if request.method == 'POST':
#         # DB 커넥션, 게시글 데이터 조회
#         conn = get_db_connection()
#         cursor = conn.cursor(cursor_factory=DictCursor)
#         today = date.today()

#         cursor.execute("""
#             SELECT title, content, title || ' ' || content AS full_text
#             FROM board.posts
#             WHERE created_at::date = %s
#         """, (today,))
#         rows = cursor.fetchall()
#         cursor.close()
#         conn.close()

#         stock_df = pd.read_csv("C:/Users/azureuser/Desktop/firstproject_flask/data/skhynix_2024_2025.csv", parse_dates=['Date'])

#         today_ts = pd.Timestamp(today)

#         today_data = stock_df[stock_df['Date'] == today_ts]
#         if not today_data.empty:
#             current_price = int(today_data.iloc[0]['Close'])
#             current_volume = int(today_data.iloc[0]['Volume'])

#         recent_df = stock_df.sort_values('Date').tail(7)
#         chart_data = {
#             'dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
#             'closes': recent_df['Close'].tolist()
#         }

#         if not rows:
#             sentiment_label = "데이터 부족"
#             prediction = "예측할 게시글이 없습니다."
#             post_sentiments = []
#         else:
#             positive_scores = []
#             negative_scores = []

#             for row in rows:
#                 post_result = predict_sentiment(row['full_text'])
#                 label = post_result['label']
#                 score = post_result['score']

#                 post_sentiments.append({
#                     "title": row['title'],
#                     "content": row['content'],
#                     "label": label,
#                     "score": score
#                 })

#                 if label == "긍정":
#                     positive_scores.append(score)
#                 elif label == "부정":
#                     negative_scores.append(score)

#             session['post_sentiments'] = post_sentiments

#             avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
#             avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0

#             # 감성 라벨 개수도 고려 (긍정, 부정 개수 비교)
#             positive_count = len(positive_scores)
#             negative_count = len(negative_scores)

#             if positive_count == 0 and negative_count == 0:
#                 sentiment_label = "중립"
#                 sentiment_score = 0
#                 prediction = (
#                     "현재 감성 분석 결과가 뚜렷하지 않아 확실한 예측을 드리기 어려운 상태입니다. "
#                     "시장에서 다양한 변수가 작용할 수 있으니 신중한 판단 부탁드립니다."
#                 )
#             else:
#                 # 평균 점수와 개수 두 가지를 모두 반영해 최종 결정 (가중치 등은 조절 가능)
#                 if (avg_positive * positive_count) > (avg_negative * negative_count):
#                     sentiment_label = "긍정"
#                     sentiment_score = avg_positive
#                     prediction = "내일 주가가 상승할 가능성이 상대적으로 높아 보입니다."
#                 elif (avg_negative * negative_count) > (avg_positive * positive_count):
#                     sentiment_label = "부정"
#                     sentiment_score = avg_negative
#                     prediction = "내일 주가가 하락할 가능성이 다소 있어 보입니다."
#                 else:
#                     sentiment_label = "중립"
#                     sentiment_score = 0
#                     prediction = (
#                         "현재 감성 분석 결과가 뚜렷하지 않아 확실한 예측을 드리기 어려운 상태입니다. "
#                         "시장에서 다양한 변수가 작용할 수 있으니 신중한 판단 부탁드립니다."
#                     )

#     return render_template(
#         'stock_predict.html',
#         prediction=prediction,
#         sentiment=sentiment_label,
#         score=sentiment_score,
#         current_price=current_price,
#         current_volume=current_volume,
#         chart_data=chart_data
#     )


# ===== "/stock_predict" 라우트 =====
@app.route('/stock_predict', methods=['GET', 'POST'])
def stock_predict():
    if request.method == 'POST':
        # 게시글 불러오기
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        today = date.today()

        cur.execute("""
            SELECT id, title, content 
            FROM board.posts 
            WHERE created_at::date = %s
        """, (today,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # 게시글 예측 수행
        results = []
        for row in rows:
            full_text = f"{row['title']} {row['content']}"
            pred = predict_stock_movement(full_text, model_pipeline)
            if pred:
                results.append({
                    'id': row['id'],
                    'title': row['title'],
                    'content': row['content'],
                    'prediction': pred['prediction'],
                    'up_prob': f"{pred['up_probability']}%",
                    'down_prob': f"{pred['down_probability']}%"
                })

        # 주가 CSV 데이터로 차트용 데이터 생성
        stock_df = pd.read_csv("C:/Users/azureuser/Desktop/firstproject_flask/data/skhynix_2024_2025.csv", parse_dates=['Date'])
        recent_df = stock_df.sort_values('Date').tail(7)

        chart_data = {
            'dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'closes': recent_df['Close'].tolist()
        }

        # 결과 저장 및 페이지 렌더링
        session['results'] = results
        return render_template('stock_predict.html', results=results, chart_data=chart_data)

    # GET 요청일 경우 예측 안함
    return render_template('stock_predict.html', results=[], chart_data=None)

# ===== "/stock_predict_detail" 라우트 =====
@app.route('/stock_predict_detail', methods=['GET'])
def stock_predict_detail():
    results = session.get('results', [])
    positive_count = sum(1 for item in results if item['prediction'] == '📈 상승')
    negative_count = sum(1 for item in results if item['prediction'] == '📉 하락')

    return render_template(
        'stock_predict_detail.html',
        results=results,
        positive_count=positive_count,
        negative_count=negative_count
    )



from collections import Counter


@app.route('/board')
def index():
    # 기존 index() 함수 내용 (게시글 목록)
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute("""
    SELECT id, title, author,
      created_at AT TIME ZONE 'Asia/Seoul' AS created_at,
      view_count, like_count
    FROM board.posts
    ORDER BY created_at DESC
""")
    posts = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', posts=posts)

@app.route('/stock')
def stock_dashboard():
    return render_template('stock_dashboard.html')  # 아래 3번의 html 파일

@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL')
    days = request.args.get('days', '5')
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period=f"{days}d")
        if hist.empty:
            return jsonify({"error": f"No data found for ticker {ticker}"}), 404

        # 필요한 컬럼만 추출
        result = hist[['Open', 'Close', 'High', 'Low', 'Volume']].reset_index()
        return jsonify(result.to_dict(orient='records'))
    except Exception as e:
        print("❌ 오류 발생:", e)
        return jsonify({"error": "500 INTERNAL SERVER ERROR", "details": str(e)}), 500


if __name__ == '__main__':
    app.debug = True
    app.run()

