from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from io import BytesIO
import numpy as np
import cv2
import os
import time
import uuid

# 导入你的引擎
from film_engine import process_style_v2_with_progress

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'film_lab_2024'
CORS(app)
# 注意：一定要设置 async_mode='eventlet' 否则进度条会卡住不更新
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

MAX_PIXELS = 4_000_000  
MAX_DIMENSION = 2400    
OUTPUT_QUALITY = 95

def smart_resize(img):
    h, w = img.shape[:2]
    scale = 1.0
    if h * w > MAX_PIXELS:
        scale = min(scale, np.sqrt(MAX_PIXELS / (h * w)))
    if max(h, w) > MAX_DIMENSION:
        scale = min(scale, MAX_DIMENSION / max(h, w))
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale
    return img, 1.0

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
        
        # 1. 发送分析开始信号
        socketio.emit('progress_update', {'session_id': session_id, 'progress': 10, 'message': '读取图像数据...'})
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return jsonify({"error": "解析失败"}), 400

        h, w = img.shape[:2]
        resized_img, scale = smart_resize(img)
        f_h, f_w = resized_img.shape[:2]

        # 2. 推送分析报告
        socketio.emit('image_analysis', {
            "session_id": session_id,
            "original": {"width": w, "height": h},
            "processed": {"width": f_w, "height": f_h},
            "will_resize": scale < 1.0,
            "estimated_time": round((f_w * f_h / 1e6) * 2.5, 1)
        })

        # 3. 定义适配 film_engine.py 的回调函数
        # 你的引擎调用格式是: callback(style_name, progress, message)
        def engine_callback(style_name, progress, message):
            socketio.emit('progress_update', {
                'session_id': session_id,
                'progress': progress,
                'message': message
            })
            socketio.sleep(0) # 释放控制权让 Socket 发送

        grain_map = {'off': 0.0, 'low': 0.5, 'normal': 1.0, 'high': 1.8}
        
        # 4. 执行处理
        processed = process_style_v2_with_progress(
            resized_img, style, grain_map.get(grain_opt, 1.0), callback=engine_callback
        )

        # 5. 编码
        socketio.emit('progress_update', {'session_id': session_id, 'progress': 95, 'message': '生成高清图片...'})
        _, buffer = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_QUALITY])
        
        res = send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')
        res.headers['X-Processing-Time'] = str(round(time.time() - start_time, 2))
        return res

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
