from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
from io import BytesIO
import numpy as np
import cv2
import os

# 注意：请确保你的算法文件名是 film_engine.py
from film_engine import process_style_v2

app = Flask(__name__, template_folder='templates')
CORS(app)

# 限制图片最大边长，防止 Vercel 内存溢出或处理超时
MAX_IMAGE_SIZE = 2500

@app.route('/')
def index():
    """渲染前端页面"""
    return render_template('index.html')

@app.route('/health')
def health():
    return {"status": "ok"}, 200

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # 1. 字段检查：前端 index.html 发送的是 'file'
        if 'file' not in request.files:
            return jsonify({"error": "No image provided (expected field 'file')"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # 2. 获取并处理参数
        style = request.form.get('style', 'fuji')
        grain_opt = request.form.get('grain', 'normal')

        # 将前端的字符串 (low/normal/high) 映射为算法需要的数字
        grain_map = {
            'off': 0.0,
            'low': 0.5,
            'normal': 1.0,
            'high': 1.8
        }
        grain_scale = grain_map.get(grain_opt, 1.0)

        # 3. 读取图片
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # 4. 性能保护：缩放过大的图片
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 5. 调用你的算法引擎
        # 确保 film_engine.py 里的 process_style_v2 接收这两个参数
        processed_img = process_style_v2(img, style, grain_scale=grain_scale)

        # 6. 将结果转换为字节流
        _, buffer = cv2.imencode('.png', processed_img)
        io_buf = BytesIO(buffer)

        return send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=False
        )

    except Exception as e:
        # 在 Vercel 的 Logs 面板可以查看到具体报错
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # 这里的配置仅用于本地测试
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
