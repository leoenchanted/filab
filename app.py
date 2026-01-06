from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from io import BytesIO
import numpy as np
import cv2
import os
import time
import uuid

# 假设 film_engine.py 在同级目录
from film_engine import process_style_v2_with_progress

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'film_lab_secret_2024'
CORS(app, resources={r"/*": {"origins": "*"}})

# 使用 eventlet 提高并发性能
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ==================== 配置 ====================
MAX_PIXELS = 4_000_000  
MAX_DIMENSION = 2400    
OUTPUT_QUALITY = 95
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'heic'}

# ==================== 工具 ====================
def progress_callback(session_id, step, progress, message):
    socketio.emit('progress_update', {
        'session_id': session_id,
        'step': step,
        'progress': progress,
        'message': message
    })
    time.sleep(0.01) # 给予 Socket 发送的小间隙

def smart_resize(img):
    h, w = img.shape[:2]
    scale = 1.0
    if h * w > MAX_PIXELS:
        scale = min(scale, np.sqrt(MAX_PIXELS / (h * w)))
    if max(h, w) > MAX_DIMENSION:
        scale = min(scale, MAX_DIMENSION / max(h, w))
    
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return img, 1.0

# ==================== 路由 ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        session_id = request.form.get('session_id')
        style = request.form.get('style', 'fuji')
        grain_opt = request.form.get('grain', 'normal')
        file = request.files['file']

        start_time = time.time()
        
        # 1. 读取与初步分析 (发送进度 10%)
        progress_callback(session_id, 'analyze', 10, '正在读取并分析图像...')
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "图片格式解析失败"}), 400

        h, w = img.shape[:2]
        
        # 2. 模拟分析逻辑并立即推送到前端面板
        resized_img, scale = smart_resize(img)
        final_h, final_w = resized_img.shape[:2]
        est_time = round((final_w * final_h / 1e6) * 2.5, 1)
        
        # 发送特定的分析结果事件
        socketio.emit('image_analysis', {
            "session_id": session_id,
            "original": {"width": w, "height": h},
            "processed": {"width": final_w, "height": final_h},
            "will_resize": scale < 1.0,
            "estimated_time": est_time
        })

        # 3. 尺寸优化 (进度 20%)
        progress_callback(session_id, 'resize', 20, '正在优化分辨率...' if scale < 1.0 else '保持原始分辨率...')
        
        # 4. 调用胶片引擎 (进度 25% - 90%)
        grain_map = {'off': 0.0, 'low': 0.5, 'normal': 1.0, 'high': 1.8}
        
        def internal_cb(step, prog, msg):
            progress_callback(session_id, step, prog, msg)

        processed = process_style_v2_with_progress(
            resized_img, style, grain_map.get(grain_opt, 1.0), callback=internal_cb
        )

        # 5. 编码与返回 (进度 95%)
        progress_callback(session_id, 'encode', 95, '正在封装高清成片...')
        _, buffer = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_QUALITY])
        
        processing_time = round(time.time() - start_time, 2)
        io_buf = BytesIO(buffer.tobytes())
        
        response = send_file(io_buf, mimetype='image/jpeg')
        response.headers['X-Processing-Time'] = str(processing_time)
        return response

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
