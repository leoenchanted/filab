from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
from io import BytesIO
import numpy as np
import cv2
import os
from functools import lru_cache

from film_engine import process_style_v2

app = Flask(__name__, template_folder='templates')
CORS(app)

# ==================== 配置参数 ====================

# 内存优化: 1600px 是画质和内存的黄金平衡点
MAX_IMAGE_SIZE = 1600

# 输出质量: 95 是肉眼无损的最佳值
OUTPUT_QUALITY = 95

# 支持的图片格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# ==================== 工具函数 ====================

def allowed_file(filename):
    """检查文件扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def smart_resize(img, max_size=MAX_IMAGE_SIZE):
    """
    智能缩放:
    - 如果图片小于 max_size,不缩放
    - 如果大于,按比例缩小
    - 使用 INTER_AREA 算法(缩小时最佳)
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def optimize_image_memory(img):
    """
    [新增] 内存优化预处理
    - 检测异常大的图片
    - 提前释放不必要的内存
    """
    h, w = img.shape[:2]
    pixel_count = h * w
    
    # 如果超过 4K 分辨率 (800万像素),强制缩小
    if pixel_count > 8_000_000:
        scale = np.sqrt(8_000_000 / pixel_count)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img

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
        # 1. 参数验证
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "文件名为空"}), 400
        
        # [新增] 文件格式检查
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"不支持的文件格式。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # 2. 获取参数
        style = request.form.get('style', 'fuji')
        grain_opt = request.form.get('grain', 'normal')
        
        # 颗粒强度映射
        grain_map = {
            'off': 0.0,     # 无颗粒
            'low': 0.5,     # 低颗粒
            'normal': 1.0,  # 标准颗粒
            'high': 1.8     # 高颗粒
        }
        grain_scale = grain_map.get(grain_opt, 1.0)
        
        # 3. 读取图片 (内存安全)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # [优化] 立即释放原始字节
        del file_bytes
        
        # 4. 格式检查
        if img is None:
            return jsonify({
                "error": "无法解析该图片格式。请尝试:\n"
                        "1. 截屏后重新上传\n"
                        "2. 手机设置 → 相机 → 格式 → 改为'兼容模式'\n"
                        "3. 使用第三方 App 转换为 JPEG"
            }), 400
        
        # 5. [新增] 内存优化预处理
        img = optimize_image_memory(img)
        
        # 6. 智能缩放
        img = smart_resize(img, MAX_IMAGE_SIZE)
        
        # 7. 执行胶片处理
        processed_img = process_style_v2(img, style, grain_scale=grain_scale)
        
        # [优化] 释放原图内存
        del img
        
        # 8. 高质量编码
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), OUTPUT_QUALITY,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,  # 启用优化
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1  # 渐进式 JPEG (更快加载)
        ]
        
        success, buffer = cv2.imencode('.jpg', processed_img, encode_params)
        
        if not success:
            return jsonify({"error": "图片编码失败"}), 500
        
        # [优化] 释放处理后的图片
        del processed_img
        
        # 9. 返回结果
        io_buf = BytesIO(buffer.tobytes())
        io_buf.seek(0)
        
        return send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'film_{style}_{grain_opt}.jpg'
        )
    
    except MemoryError:
        return jsonify({
            "error": "图片过大,服务器内存不足。请尝试:\n"
                    "1. 压缩图片后重新上传\n"
                    "2. 使用截图代替原图"
        }), 507
    
    except Exception as e:
        # 打印完整错误到日志
        import traceback
        print("=" * 50)
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        print("=" * 50)
        
        return jsonify({
            "error": f"处理失败: {str(e)}"
        }), 500

# ==================== 健康检查 ====================

@app.route('/health', methods=['GET'])
def health_check():
    """用于 Vercel/Docker 的健康检查"""
    return jsonify({
        "status": "healthy",
        "max_size": MAX_IMAGE_SIZE,
        "quality": OUTPUT_QUALITY
    }), 200

# ==================== 启动 ====================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=port,
        threaded=True  # 启用多线程
    )
