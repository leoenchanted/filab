from flask import Flask, request, send_file, jsonify, render_template, send_from_directory, make_response, session, redirect, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from io import BytesIO
import numpy as np
import cv2
import os
import time
import uuid
import json
import requests
from datetime import datetime, timedelta

from film_engine import process_style_v2_with_progress

app = Flask(__name__, template_folder='templates', static_folder='static')
# ★★★ 这个KEY用于加密Session，必须保留 ★★★
app.config['SECRET_KEY'] = 'film_lab_pro_secret_key_888' 
CORS(app)

# ==================== Cloudflare 真实 IP 逻辑 ====================
def get_real_ip():
    if request.headers.get('CF-Connecting-IP'):
        return request.headers.get('CF-Connecting-IP')
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr

# ==================== 初始化限流器 ====================
limiter = Limiter(
    key_func=get_real_ip,
    app=app,
    default_limits=["2000 per day", "200 per hour"],
    storage_uri="memory://"
)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ==================== 配置 ====================
MAX_PIXELS = 12_000_000
MAX_DIMENSION = 2400
OUTPUT_QUALITY = 95
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'heic'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_FILE = os.path.join(BASE_DIR, 'visitor_data.json')

# ==================== IP 位置查询 ====================
def get_ip_location(ip):
    location = "Unknown"
    cf_country = request.headers.get('CF-IPCountry')
    if cf_country:
        location = f"{cf_country} (Cloudflare)"
    if ip.startswith("127.") or ip.startswith("192.168.") or ip == "localhost":
        return "Localhost / Intranet"
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}?lang=zh-CN", timeout=2)
        if response.status_code == 200 and response.json().get('status') == 'success':
            data = response.json()
            location = f"{data.get('country')} {data.get('regionName')} {data.get('city')}"
    except:
        pass
    return location

# ==================== 统计功能 ====================
def init_stats_file():
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"total_unique_visitors": 0, "history": []}, f, ensure_ascii=False, indent=2)

def update_visitor_activity(visitor_id, ip, is_new_cookie=False):
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"total_unique_visitors": 0, "history": []}

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        found_record = None
        for record in data['history']:
            if record.get('id') == visitor_id:
                found_record = record
                break
        
        if found_record:
            # 老访客：更新
            data['history'].remove(found_record)
            found_record['visits'] = found_record.get('visits', 1) + 1
            found_record['last_seen'] = current_time
            if ip != found_record.get('ip'):
                found_record['ip'] = ip
                found_record['location'] = get_ip_location(ip)
            data['history'].insert(0, found_record)
        else:
            # 新访客：插入
            if is_new_cookie or not found_record:
                data["total_unique_visitors"] += 1
            
            location = get_ip_location(ip)
            new_record = {
                "id": visitor_id,
                "ip": ip,
                "location": location,
                "first_seen": current_time,
                "last_seen": current_time,
                "visits": 1
            }
            data["history"].insert(0, new_record)

        if len(data["history"]) > 5000:
            data["history"] = data["history"][:5000]

        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return data["total_unique_visitors"]
    except Exception as e:
        print(f"Stats Error: {e}")
        return data.get("total_unique_visitors", 0)

def get_total_count():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get("total_unique_visitors", 0)
    except: pass
    return 0

init_stats_file()

# ==================== 工具函数 ====================
def smart_resize_by_pixels(img):
    h, w = img.shape[:2]
    total_pixels = h * w
    scale = 1.0
    if total_pixels > MAX_PIXELS: scale = min(scale, np.sqrt(MAX_PIXELS / total_pixels))
    if max(h, w) > MAX_DIMENSION: scale = min(scale, MAX_DIMENSION / max(h, w))
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale
    return img, 1.0

def progress_callback(session_id, step, progress, message):
    socketio.emit('progress_update', {'session_id': session_id, 'step': step, 'progress': progress, 'message': message}, namespace='/')

# ==================== 路由 ====================

@app.route('/')
def index():
    visitor_id = request.cookies.get('visitor_id')
    user_ip = get_real_ip()
    
    if visitor_id is None:
        new_id = str(uuid.uuid4())
        total = update_visitor_activity(new_id, user_ip, is_new_cookie=True)
        resp = make_response(render_template('index.html', total_visitors="{:,}".format(total)))
        resp.set_cookie('visitor_id', new_id, expires=datetime.now() + timedelta(days=365))
        return resp
    else:
        total = update_visitor_activity(visitor_id, user_ip, is_new_cookie=False)
        return render_template('index.html', total_visitors="{:,}".format(total))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico')

# ==================== ★★★ 管理员后台逻辑 ★★★ ====================
@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    error = None
    
    # 1. 如果点击了登出，清除 Session，重定向回 admin (触发登录框)
    if request.args.get('action') == 'logout':
        session.pop('is_admin', None)
        return redirect(url_for('admin_panel'))

    # 2. 如果是 POST 请求，说明用户提交了表单
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 验证账号密码
        if username == 'zzh' and password == '060312':
            session['is_admin'] = True  # 登录成功，写入 Session
            return redirect(url_for('admin_panel')) # 刷新页面
        else:
            error = "ACCESS DENIED: INVALID CREDENTIALS" # 错误提示

    # 3. 检查 Session (核心判断)
    # 如果 Session 里没有 is_admin，说明没登录 -> 显示登录框
    if not session.get('is_admin'):
        return render_template('admin.html', show_login=True, error=error)

    # 4. 如果代码跑到这里，说明 is_admin 为 True (已登录) -> 显示表格
    data = {"total_unique_visitors": 0, "history": []}
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
    except: pass
    
    formatted_total = "{:,}".format(data.get("total_unique_visitors", 0))
    return render_template('admin.html', show_login=False, total_visitors=formatted_total, history=data.get("history", []))
# =================================================================

@app.route('/process', methods=['POST'])
@limiter.limit("5 per minute")
def process_image():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        session_id = request.form.get('session_id', 'unknown')
        style = request.form.get('style', 'fuji')
        
        start_time = time.time()
        progress_callback(session_id, 'loading', 10, 'Receiving...')
        
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return jsonify({"error": "Bad Image"}), 400
        
        img, scale = smart_resize_by_pixels(img)
        progress_callback(session_id, 'processing', 30, 'Developing...')
        
        processed_img = process_style_v2_with_progress(
            img, style, 0.0, callback=lambda s, p, m: progress_callback(session_id, s, 30 + (p * 0.6), m)
        )
        del img
        
        progress_callback(session_id, 'encoding', 95, 'Scanning...')
        success, buffer = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_QUALITY])
        if not success: return jsonify({"error": "Encoding Error"}), 500
        
        io_buf = BytesIO(buffer.tobytes())
        io_buf.seek(0)
        progress_callback(session_id, 'complete', 100, 'Done')
        
        response = send_file(io_buf, mimetype='image/jpeg', as_attachment=True, download_name='film.jpg')
        response.headers['X-Processing-Time'] = str(round(time.time() - start_time, 2))
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "RATE LIMIT EXCEEDED"}), 429

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)