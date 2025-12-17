from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import io
from film_engine import process_style_v2

# 指向 templates 文件夹
app = Flask(__name__, template_folder='templates')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 

# 新增：访问根目录直接返回网页
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    # ... (保持原来的代码不变) ...
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    style = request.form.get('style', 'fuji')
    grain_opt = request.form.get('grain', 'normal')
    
    scale_map = {'off': 0.0, 'low': 0.5, 'normal': 1.0, 'high': 1.5, 'extreme': 2.5}
    grain_scale = scale_map.get(grain_opt, 1.0)
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: return jsonify({"error": "Invalid Image"}), 400
        processed_img = process_style_v2(img, style, grain_scale)
        is_success, buffer = cv2.imencode(".png", processed_img)
        if not is_success: return jsonify({"error": "Encode failed"}), 500
        return send_file(io.BytesIO(buffer), mimetype='image/png')
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)