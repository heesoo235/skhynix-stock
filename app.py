from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import psycopg2
from psycopg2.extras import DictCursor
import os
from datetime import datetime
from dotenv import load_dotenv
from markupsafe import Markup
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from finbert import predict_sentiment 
from flask import render_template, request, redirect, url_for, session
from datetime import date

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

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

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute(
        "INSERT INTO board.posts (title, content, author) VALUES (%s, %s, %s) RETURNING id",
        (title, content, author)
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
    cursor.execute('UPDATE board.posts SET view_count = view_count + 1 WHERE id = %s', (post_id,))
    cursor.execute('SELECT * FROM board.posts WHERE id = %s', (post_id,))
    post = cursor.fetchone()
    if post is None:
        cursor.close()
        conn.close()
        flash('게시글이 없습니다.')
        return redirect(url_for('index'))

    cursor.execute('SELECT * FROM board.comments WHERE post_id = %s ORDER BY created_at', (post_id,))
    comments = cursor.fetchall()
    cursor.close()
    conn.close()

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
#     sentiment = None
#     if request.method == 'POST':
#         # 예시: 게시판 오늘 글을 다 불러와서 감성분석 후 예측
#         conn = get_db_connection()
#         cursor = conn.cursor(cursor_factory=DictCursor)
#         from datetime import date
#         today = date.today()
#         cursor.execute("SELECT content FROM board.posts WHERE created_at::date = %s", (today,))
#         rows = cursor.fetchall()
#         cursor.close()
#         conn.close()
#         combined_text = " ".join([row['content'] for row in rows])
#         # 간단 감성 분석/예측 예시 (여기서 실제 ML 모델 연결 가능)
#         sentiment = "positive" if "좋다" in combined_text else "neutral"
#         prediction = "내일 주가가 상승할 가능성이 높음!" if sentiment == "positive" else "변동성 주의!"
#     return render_template('stock_predict.html', prediction=prediction, sentiment=sentiment)


@app.route('/stock_predict', methods=['GET', 'POST'])
def stock_predict():
    prediction = None
    sentiment_label = None
    sentiment_score = None
    post_sentiments = []
    current_price = None
    current_volume = None
    chart_data = None

    if request.method == 'POST':
        # DB 커넥션, 게시글 데이터 조회
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        today = date.today()

        cursor.execute("""
            SELECT title, content, title || ' ' || content AS full_text
            FROM board.posts
            WHERE created_at::date = %s
        """, (today,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        stock_df = pd.read_csv("C:/Users/azureuser/Desktop/firstproject_flask/data/skhynix_2024_2025.csv", parse_dates=['Date'])

        today_ts = pd.Timestamp(today)

        today_data = stock_df[stock_df['Date'] == today_ts]
        if not today_data.empty:
            current_price = int(today_data.iloc[0]['Close'])
            current_volume = int(today_data.iloc[0]['Volume'])

        recent_df = stock_df.sort_values('Date').tail(7)
        chart_data = {
            'dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'closes': recent_df['Close'].tolist()
        }

        if not rows:
            sentiment_label = "데이터 부족"
            prediction = "예측할 게시글이 없습니다."
            post_sentiments = []
        else:
            positive_scores = []
            negative_scores = []

            for row in rows:
                post_result = predict_sentiment(row['full_text'])
                label = post_result['label']
                score = post_result['score']

                post_sentiments.append({
                    "title": row['title'],
                    "content": row['content'],
                    "label": label,
                    "score": score
                })

                if label == "긍정":
                    positive_scores.append(score)
                elif label == "부정":
                    negative_scores.append(score)

            session['post_sentiments'] = post_sentiments

            avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
            avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0

            # 감성 라벨 개수도 고려 (긍정, 부정 개수 비교)
            positive_count = len(positive_scores)
            negative_count = len(negative_scores)

            if positive_count == 0 and negative_count == 0:
                sentiment_label = "중립"
                sentiment_score = 0
                prediction = (
                    "현재 감성 분석 결과가 뚜렷하지 않아 확실한 예측을 드리기 어려운 상태입니다. "
                    "시장에서 다양한 변수가 작용할 수 있으니 신중한 판단 부탁드립니다."
                )
            else:
                # 평균 점수와 개수 두 가지를 모두 반영해 최종 결정 (가중치 등은 조절 가능)
                if (avg_positive * positive_count) > (avg_negative * negative_count):
                    sentiment_label = "긍정"
                    sentiment_score = avg_positive
                    prediction = "내일 주가가 상승할 가능성이 상대적으로 높아 보입니다."
                elif (avg_negative * negative_count) > (avg_positive * positive_count):
                    sentiment_label = "부정"
                    sentiment_score = avg_negative
                    prediction = "내일 주가가 하락할 가능성이 다소 있어 보입니다."
                else:
                    sentiment_label = "중립"
                    sentiment_score = 0
                    prediction = (
                        "현재 감성 분석 결과가 뚜렷하지 않아 확실한 예측을 드리기 어려운 상태입니다. "
                        "시장에서 다양한 변수가 작용할 수 있으니 신중한 판단 부탁드립니다."
                    )

    return render_template(
        'stock_predict.html',
        prediction=prediction,
        sentiment=sentiment_label,
        score=sentiment_score,
        current_price=current_price,
        current_volume=current_volume,
        chart_data=chart_data
    )


from collections import Counter

import pprint
@app.route('/stock_predict_detail')
def stock_predict_detail():
    post_sentiments = session.get('post_sentiments', [])
    
    print("[DEBUG] post_sentiments in session:", post_sentiments)

    prediction = session.get('prediction', None)
    sentiment_label = session.get('sentiment_label', None)
    current_price = session.get('current_price', None)
    current_volume = session.get('current_volume', None)
    chart_data = session.get('chart_data', None)

    label_counts = Counter(post['label'] for post in post_sentiments if post['label'] in ['긍정', '부정'])
    positive_count = label_counts.get('긍정', 0)
    negative_count = label_counts.get('부정', 0)

    return render_template(
        'stock_predict_detail.html',
        post_sentiments=post_sentiments,
        positive_count=positive_count,
        negative_count=negative_count,
        prediction=prediction,
        sentiment=sentiment_label,
        current_price=current_price,
        current_volume=current_volume,
        chart_data=chart_data
    )


@app.route('/board')
def index():
    # 기존 index() 함수 내용 (게시글 목록)
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute("SELECT id, title, author, created_at, view_count, like_count FROM board.posts ORDER BY created_at DESC")
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

