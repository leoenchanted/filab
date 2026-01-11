from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from io import BytesIO
import numpy as np
import cv2
import os
import time
import uuid

from film_engine import process_style_v2_with_progress

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'film_lab_pro_secret'
CORS(app)

# ==================== 1. 初始化限流器 (新增) ====================
# 默认限制：每个IP每天200次，每小时50次 (防止有人疯狂刷页面)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1000 per day", "100 per hour"],
    storage_uri="memory://"
)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ==================== 配置参数 ====================
# 已改为专业画质参数
MAX_PIXELS = 12_000_000
MAX_DIMENSION = 2400
OUTPUT_QUALITY = 95
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'heic'}

# ==================== 工具函数 ====================

def smart_resize_by_pixels(img):
    h, w = img.shape[:2]
    total_pixels = h * w
    scale = 1.0
    if total_pixels > MAX_PIXELS:
        scale = min(scale, np.sqrt(MAX_PIXELS / total_pixels))
    if max(h, w) > MAX_DIMENSION:
        scale = min(scale, MAX_DIMENSION / max(h, w))
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return img, 1.0

def progress_callback(session_id, step, progress, message):
    socketio.emit('progress_update', {
        'session_id': session_id,
        'step': step,
        'progress': progress,
        'message': message
    }, namespace='/')

# ==================== 路由 ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico')

@app.route('/process', methods=['POST'])
@limiter.limit("5 per minute")  # ★★★ 核心：限制胶片处理接口，每分钟每IP只能调5次 ★★★
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id', 'unknown')
        style = request.form.get('style', 'fuji')
        # grain 在后端不处理，设为 off
        grain_opt = request.form.get('grain', 'off')
        
        start_time = time.time()
        
        # 1. Loading
        progress_callback(session_id, 'loading', 10, 'Receiving...')
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Bad Image"}), 400
        
        # 2. Resizing (Safety Net)
        img, scale = smart_resize_by_pixels(img)
        
        # 3. Processing
        progress_callback(session_id, 'processing', 30, 'Developing...')
        
        def internal_callback(step, progress, message):
            mapped_progress = 30 + (progress * 0.6) 
            progress_callback(session_id, step, mapped_progress, message)
        
        processed_img = process_style_v2_with_progress(
            img, style, 0.0, callback=internal_callback
        )
        del img
        
        # 4. Encoding
        progress_callback(session_id, 'encoding', 95, 'Scanning...')
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_QUALITY, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
        success, buffer = cv2.imencode('.jpg', processed_img, encode_params)
        
        if not success: return jsonify({"error": "Encoding Error"}), 500
        
        io_buf = BytesIO(buffer.tobytes())
        io_buf.seek(0)
        
        progress_callback(session_id, 'complete', 100, 'Done')
        
        processing_time = time.time() - start_time
        response = send_file(io_buf, mimetype='image/jpeg', as_attachment=True, download_name='film.jpg')
        response.headers['X-Processing-Time'] = str(round(processing_time, 2))
        return response
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 处理限流报错 (返回 429 状态码)
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "请求太快了，请稍后再试 (Rate limit exceeded)"}), 429

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)