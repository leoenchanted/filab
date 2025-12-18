from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
from io import BytesIO
import numpy as np
import cv2
import os

from film_engine import process_style_v2

app = Flask(__name__)
CORS(app)


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
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        style = request.form.get('style', 'kodak')

        processed_img = process_style_v2(img, style)

        _, buffer = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        io_buf = BytesIO(buffer)

        return send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=False
        )

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": "Processing failed"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
