from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2
from psycopg2.extras import DictCursor
import os
from datetime import datetime
from dotenv import load_dotenv
from markupsafe import Markup

load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24)


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

@app.route('/stock_predict', methods=['GET', 'POST'])
def stock_predict():
    prediction = None
    sentiment = None
    if request.method == 'POST':
        # 예시: 게시판 오늘 글을 다 불러와서 감성분석 후 예측
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        from datetime import date
        today = date.today()
        cursor.execute("SELECT content FROM board.posts WHERE created_at::date = %s", (today,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        combined_text = " ".join([row['content'] for row in rows])
        # 간단 감성 분석/예측 예시 (여기서 실제 ML 모델 연결 가능)
        sentiment = "positive" if "좋다" in combined_text else "neutral"
        prediction = "내일 주가가 상승할 가능성이 높음!" if sentiment == "positive" else "변동성 주의!"
    return render_template('stock_predict.html', prediction=prediction, sentiment=sentiment)

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


if __name__ == '__main__':
    app.debug = True
    app.run()

