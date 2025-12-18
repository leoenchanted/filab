from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
from io import BytesIO
import numpy as np
import cv2
import os

from film_engine import process_style_v2

app = Flask(__name__, template_folder='templates')
CORS(app)

# 针对 Vercel 免费版内存优化：
# 1600px 是画质和内存的黄金平衡点（支持 2.5K 分辨率）
MAX_IMAGE_SIZE = 1600

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
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        style = request.form.get('style', 'fuji')
        grain_opt = request.form.get('grain', 'normal')

        grain_map = {'off': 0.0, 'low': 0.5, 'normal': 1.0, 'high': 1.8}
        grain_scale = grain_map.get(grain_opt, 1.0)

        # 1. 安全读取
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 2. 检查格式是否兼容 (解决手机拍摄 HEIC 问题)
        if img is None:
            return jsonify({
                "error": "无法解析该图片格式。请尝试上传截屏，或在手机设置中将格式改为‘兼容模式’。"
            }), 400

        # 3. 内存安全保护：如果是超大图，强制缩放
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            # 使用 INTER_AREA 是处理大图缩小的最佳算法
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 4. 执行胶片算法
        # 确保你的 film_engine.py 里的函数名和参数能对上
        processed_img = process_style_v2(img, style, grain_scale=grain_scale)

        # 5. 高画质输出
        # 虽然 PNG 无损，但体积大，容易导致手机下载慢。
        # 我们用高画质 JPEG (95%)，肉眼看不出区别，但速度快 5 倍。
        _, buffer = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        io_buf = BytesIO(buffer)

        return send_file(io_buf, mimetype='image/jpeg')

    except Exception as e:
        # 打印具体错误到 Vercel 日志
        print(f"Server Error: {str(e)}")
        return jsonify({"error": f"暗房处理失败: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
