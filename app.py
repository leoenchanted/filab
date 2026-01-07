from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from io import BytesIO
import numpy as np
import cv2
import os
import time
import uuid

# 引入之前的处理引擎
from film_engine import process_style_v2_with_progress

# 显式指定 static 文件夹
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'film_lab_secret_2024'
CORS(app, resources={r"/*": {"origins": "*"}})

# WebSocket 配置
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ==================== 配置参数 ====================

MAX_PIXELS = 4_000_000
MAX_DIMENSION = 2400
OUTPUT_QUALITY = 95
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'heic'}

# ==================== 工具函数 ====================

def smart_resize_by_pixels(img, max_pixels=MAX_PIXELS, max_dim=MAX_DIMENSION):
    h, w = img.shape[:2]
    total_pixels = h * w
    scale = 1.0
    
    if total_pixels > max_pixels:
        scale = min(scale, np.sqrt(max_pixels / total_pixels))
    
    if max(h, w) > max_dim:
        scale = min(scale, max_dim / max(h, w))
    
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
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        
        if file.filename == '':
            return jsonify({"error": "文件名为空"}), 400
        
        style = request.form.get('style', 'fuji')
        grain_opt = request.form.get('grain', 'normal')
        
        grain_map = {'off': 0.0, 'low': 0.5, 'normal': 1.0, 'high': 1.8}
        grain_scale = grain_map.get(grain_opt, 1.0)
        
        start_time = time.time()
        
        progress_callback(session_id, 'loading', 10, '加载底片...')
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "无法解析图片"}), 400
        
        progress_callback(session_id, 'resize', 20, '规格检查...')
        h_origin, w_origin = img.shape[:2]
        img, scale = smart_resize_by_pixels(img, MAX_PIXELS, MAX_DIMENSION)
        h_new, w_new = img.shape[:2]
        
        progress_callback(session_id, 'processing', 30, '胶片显影中...')
        
        def internal_callback(step, progress, message):
            mapped_progress = 30 + (progress * 0.6) 
            progress_callback(session_id, step, mapped_progress, message)
        
        processed_img = process_style_v2_with_progress(
            img, style, grain_scale, callback=internal_callback
        )
        
        progress_callback(session_id, 'encoding', 95, '定影输出...')
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_QUALITY, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
        success, buffer = cv2.imencode('.jpg', processed_img, encode_params)
        
        if not success:
            return jsonify({"error": "编码失败"}), 500
        
        io_buf = BytesIO(buffer.tobytes())
        io_buf.seek(0)
        
        processing_time = time.time() - start_time
        progress_callback(session_id, 'complete', 100, '✨ 完成')
        
        response = send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'film_{style}_{grain_opt}.jpg'
        )
        
        response.headers['X-Processing-Time'] = str(round(processing_time, 2))
        response.headers['X-Original-Size'] = f"{w_origin}x{h_origin}"
        response.headers['X-Processed-Size'] = f"{w_new}x{h_new}"
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)