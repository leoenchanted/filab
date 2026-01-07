from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from io import BytesIO
import numpy as np
import cv2
import os
import time
import uuid

from film_engine import process_style_v2_with_progress

app = Flask(__name__, template_folder='templates')
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def get_image_info(img):
    h, w = img.shape[:2]
    return {
        "width": w,
        "height": h,
        "pixels": w * h,
        "megapixels": round(w * h / 1_000_000, 2)
    }

def progress_callback(session_id, step, progress, message):
    """发送进度到前端"""
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

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400
        
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "无法解析图片"}), 400
        
        h, w = img.shape[:2]
        original_pixels = h * w
        
        resized_img, scale = smart_resize_by_pixels(img)
        final_h, final_w = resized_img.shape[:2]
        
        estimated_time = (final_w * final_h / 1_000_000) * 2.5
        
        return jsonify({
            "original": {"width": w, "height": h, "megapixels": round(original_pixels/1e6, 2)},
            "processed": {"width": final_w, "height": final_h, "megapixels": round(final_w*final_h/1e6, 2)},
            "will_resize": scale < 1.0,
            "resize_ratio": round(scale * 100, 1),
            "estimated_time": round(estimated_time, 1)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400
        
        file = request.files['file']
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        
        if file.filename == '':
            return jsonify({"error": "文件名为空"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"不支持的文件格式。支持: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        style = request.form.get('style', 'fuji')
        grain_opt = request.form.get('grain', 'normal')
        
        grain_map = {'off': 0.0, 'low': 0.5, 'normal': 1.0, 'high': 1.8}
        grain_scale = grain_map.get(grain_opt, 1.0)
        
        start_time = time.time()
        
        progress_callback(session_id, 'loading', 5, '读取图像数据...')
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        del file_bytes
        
        if img is None:
            return jsonify({"error": "无法解析图片格式"}), 400
        
        progress_callback(session_id, 'loading', 15, '分析图像信息...')
        original_info = get_image_info(img)
        
        progress_callback(session_id, 'resize', 20, '优化图像尺寸...')
        img, scale = smart_resize_by_pixels(img, MAX_PIXELS, MAX_DIMENSION)
        resized_info = get_image_info(img)
        
        progress_callback(session_id, 'processing', 25, '开始胶片处理...')
        
        def internal_callback(step, progress, message):
            progress_callback(session_id, step, progress, message)
        
        processed_img = process_style_v2_with_progress(
            img, style, grain_scale, callback=internal_callback
        )
        del img
        
        progress_callback(session_id, 'encoding', 90, '生成高清成片...')
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_QUALITY,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1
        ]
        
        success, buffer = cv2.imencode('.jpg', processed_img, encode_params)
        del processed_img
        
        if not success:
            return jsonify({"error": "图片编码失败"}), 500
        
        progress_callback(session_id, 'complete', 100, '✨ 冲洗完成!')
        
        io_buf = BytesIO(buffer.tobytes())
        io_buf.seek(0)
        
        processing_time = time.time() - start_time
        
        response = send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'film_{style}_{grain_opt}.jpg'
        )
        
        response.headers['X-Processing-Time'] = str(round(processing_time, 2))
        response.headers['X-Original-Size'] = f"{original_info['width']}x{original_info['height']}"
        response.headers['X-Processed-Size'] = f"{resized_info['width']}x{resized_info['height']}"
        response.headers['X-Was-Resized'] = str(scale < 1.0).lower()
        
        return response
    
    except MemoryError:
        return jsonify({"error": "图片过大,服务器内存不足"}), 507
    
    except Exception as e:
        import traceback
        print("=" * 50)
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print("=" * 50)
        return jsonify({"error": f"处理失败: {str(e)}"}), 500

# ==================== WebSocket 事件 ====================

@socketio.on('connect')
def handle_connect():
    print(f'✅ Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'❌ Client disconnected: {request.sid}')

# ==================== 健康检查 ====================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "max_pixels": MAX_PIXELS,
        "max_dimension": MAX_DIMENSION,
        "quality": OUTPUT_QUALITY,
        "websocket": "enabled"
    }), 200

# ==================== 启动 ====================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    socketio.run(
        app,
        debug=debug_mode,
        host='0.0.0.0',
        port=port,
        allow_unsafe_werkzeug=True
    )
