import os
import sys

# Configuration - Point to frontend folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'frontend/templates')
STATIC_DIR = os.path.join(BASE_DIR, 'frontend/static')
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Create directories if they don't exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

from flask import Flask, render_template, request, jsonify

# Set TensorFlow logging BEFORE import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global model variable
model = None

def load_model_eager():
    """Load model at startup (eager loading)"""
    global model
    if model is None:
        try:
            model_path = os.path.join(PROJECT_ROOT, 'Model/waste_classifier.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            print("📦 Loading AI model...")
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded and ready!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    return model

def load_model_lazy():
    """Load model on first use to avoid threading issues"""
    return load_model_eager()

# Class definitions - MUST match the model's training classes in sorted order!
CLASS_NAMES = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

CLASS_COLORS = {
    'Cardboard': '#8B4513',
    'Food Organics': '#228B22',
    'Glass': '#87CEEB',
    'Metal': '#C0C0C0',
    'Miscellaneous Trash': '#696969',
    'Paper': '#FFD700',
    'Plastic': '#FF6B6B',
    'Textile Trash': '#9370DB',
    'Vegetation': '#00AA00'
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        import numpy as np
        from PIL import Image
        
        # Validate file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate extension
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in ['jpg', 'jpeg', 'png', 'bmp']:
            return jsonify({'error': 'Invalid file format. Use JPG, PNG, or BMP'}), 400
        
        # Load model on first use
        loaded_model = load_model_lazy()
        
        # Read and preprocess image
        img = Image.open(file.stream)
        img = img.convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        predictions = loaded_model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = CLASS_NAMES[predicted_idx]
        
        # Prepare detailed results
        all_predictions = [
            {
                'class': CLASS_NAMES[i],
                'confidence': round(float(predictions[0][i]) * 100, 2)
            }
            for i in range(len(CLASS_NAMES))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        results = {
            'class': predicted_class,
            'confidence': confidence,
            'confidence_percent': round(confidence * 100, 2),
            'color': CLASS_COLORS.get(predicted_class, '#667eea'),
            'all_predictions': all_predictions
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  ♻️  RealWaste Classifier - Flask Application")
    print("="*70)
    print(f"\n  📁 Base Directory: {BASE_DIR}")
    print(f"  🤖 Model: {os.path.join(BASE_DIR, 'Model/mobilenet_v2_model.h5')}")
    print(f"  🌐 Server: http://localhost:8000")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=8000, threaded=False)
