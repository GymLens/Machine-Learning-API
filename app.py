from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uuid  # Untuk membuat ID unik
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

model = None
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(file_path):
    """
    Preprocess the image for the model.
    Resize the image to 224x224 and normalize it.
    """
    try:
        # Load image and resize it to 224x224 as required by the model
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request files:", request.files)  # Log the request files

    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    try:
        # Check if the file is present in the request
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file found in the request'}), 400
        
        # Get the uploaded image file
        image_file = request.files['image']
        print(f"Uploaded file: {image_file.filename}")  # Log the uploaded file name
        
        if image_file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty file'}), 400
        
        # Save the image to the upload folder
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        
        # Preprocess the image and make prediction
        processed_image = preprocess_image(file_path)
        predictions = model.predict(processed_image)[0]
        
        # Get the class with the highest probability
        predicted_index = np.argmax(predictions)  # Get the class with highest probability
        predicted_class_name = CLASS_NAMES[predicted_index]  # Get the class name
        confidence = predictions[predicted_index]  # Get the probability
        
        # Generate a unique ID
        unique_id = str(uuid.uuid4())

        # Return the result as a response
        return jsonify({
            'message': "Model is predicted successfully.",
            'data': {
                'id': unique_id,
                'result': predicted_class_name,
                'confidenceScore': round(confidence * 100, 2),  # Convert to percentage
                'isAboveThreshold': bool(confidence >= 0.5),  # Ensure boolean type
                'createdAt': datetime.now().isoformat() + 'Z'
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error processing request: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
