from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
MODEL_PATH = './gym_model_densenet121_9958.h5'
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")

CLASS_NAMES = [
    'Bench Press', 'Dip Bar', 'Dumbells', 'Elliptical Machine',
    'KettleBell', 'Lat Pulldown', 'Leg Press Machine',
    'PullBar', 'Recumbent Bike', 'Stair Climber',
    'Swiss Ball', 'Treadmill'
]

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def preprocess_image(file_path):
    """
    Preprocess the image for the model
    """
    try:
        img = load_img(file_path, target_size=(224, 224))  # Sesuaikan dengan input model Anda
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


@app.route('/predict', methods=['POST'])
def predict():
    print("Received request files:", request.files)  # Tambahkan log ini
    if not model:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    try:
        # Check if file is present in the request
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file found in the request'}), 400
        
        # Save the uploaded file
        image_file = request.files['image']
        print(f"Uploaded file: {image_file.filename}")  # Log nama file yang diupload
        if image_file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty file'}), 400
        
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        
        # Preprocess and predict
        processed_image = preprocess_image(file_path)
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]  # Ambil class dengan probabilitas tertinggi
        predicted_class_name = CLASS_NAMES[predicted_index]  # Mendapatkan nama kelas
        
        return jsonify({
            'predicted_class': predicted_class_name  # Mengembalikan hanya nama kelas
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error processing request: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
