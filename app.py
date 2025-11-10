
from flask import Flask, render_template, request, jsonify
import os
from model import create_or_load_model, predict_emotion
import sqlite3
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = create_or_load_model()

def init_db():
    conn = sqlite3.connect('emotion_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY, name TEXT, image_path TEXT, emotion TEXT, confidence REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Anonymous')
    image_path = None

    if 'image' in request.files:
        image_file = request.files['image']
        if image_file and image_file.filename != '':
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            image_path = os.path.join(UPLOAD_FOLDER, f"{name}_{timestamp}.jpg")
            image_file.save(image_path)

    elif 'image_b64' in request.form:
        image_b64 = request.form['image_b64']
        try:
            image_data = base64.b64decode(image_b64.split(',')[1])
            image = Image.open(BytesIO(image_data))
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            image_path = os.path.join(UPLOAD_FOLDER, f"{name}_{timestamp}.jpg")
            image.save(image_path)
        except Exception as e:
            return jsonify({'error': 'Invalid image data'}), 400
    else:
        return jsonify({'error': 'No image provided'}), 400

    try:
        emotion, confidence = predict_emotion(model, image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # --- Log to DB ---
    conn = sqlite3.connect('emotion_logs.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (name, image_path, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
              (name, image_path, emotion, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()
    
    return jsonify({'emotion': emotion, 'confidence': f"{confidence:.2f}"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)