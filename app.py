import os
import sys
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tf_keras as keras
import numpy as np
import cv2
import base64

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        print(f"Running from _MEIPASS: {base_path}")
    except Exception:
        base_path = os.path.abspath(".")
        print(f"Running in development mode at: {base_path}")
    return os.path.join(base_path, relative_path)

# --- Configuration ---
MODEL_PATH = resource_path("keras_model.h5")
LABELS_PATH = resource_path("labels.txt")
STATIC_DIR = resource_path("web_app")

print(f"Checking for static folder at: {STATIC_DIR}")
if os.path.exists(STATIC_DIR):
    print(f"Confirmed: {STATIC_DIR} exists.")
    print(f"Contents of {STATIC_DIR}: {os.listdir(STATIC_DIR)}")
else:
    print(f"CRITICAL ERROR: {STATIC_DIR} DOES NOT EXIST.")

# Create Flask app with static folder mapping
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)

# Load labels
try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception:
    class_names = ["0 Pituitary", "1 Meningioma", "2 Glioma", "3 No Tumor"]

# Load model
print("Loading AI Model (this may take a moment)...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# --- Web Routes ---
@app.route('/')
def index():
    print("Serving index.html from static folder.")
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Convert base64 to OpenCV image (BGR by default)
        encoded_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # --- PREPROCESSING FIX ---
        # 1. Convert BGR to RGB (Models usually expect RGB, OpenCV decodes to BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Center crop to square (Maintains aspect ratio, prevents squashing)
        h, w = img_rgb.shape[:2]
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        img_cropped = img_rgb[start_h:start_h + min_dim, start_w:start_w + min_dim]
        
        # 3. Resize to 224x224
        img_resized = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_AREA)
        
        # 4. Prepare for model (Normalize to [-1, 1])
        image_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1
        
        # Predict
        prediction = model.predict(image_array, verbose=0)
        
        results = []
        for i, prob in enumerate(prediction[0]):
            name = class_names[i][2:] if len(class_names[i]) > 2 else class_names[i]
            results.append({
                'label': name,
                'confidence': float(prob)
            })
            
        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def open_browser():
    # Only open browser if not in debug mode to avoid double opens
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    print("----------------------------------------")
    print("Tumor Classification App is starting...")
    print("Opening browser at http://localhost:5000")
    print("----------------------------------------")
    
    # Start browser with a small delay
    Timer(1.5, open_browser).start()
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)
