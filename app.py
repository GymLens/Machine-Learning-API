from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from datetime import datetime
import uuid
from service_account_token_generator import generate_access_token

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
MODEL_PATH = './gym_model_densenet121_9958.h5'
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
    try:
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file found in the request'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty file'}), 400
        
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        
        processed_image = preprocess_image(file_path)
        predictions = model.predict(processed_image)[0]
        
        predicted_index = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_index]
        confidence = predictions[predicted_index]
        
        unique_id = str(uuid.uuid4())

        return jsonify({
            'message': "Model is predicted successfully.",
            'data': {
                'id': unique_id,
                'result': predicted_class_name,
                'confidenceScore': round(confidence * 100, 2),
                'isAboveThreshold': bool(confidence >= 0.5),
                'createdAt': datetime.now().isoformat() + 'Z'
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error processing request: {str(e)}"
        }), 500

@app.route('/generate-token', methods=['GET'])
def generate_token_endpoint():
    SERVICE_ACCOUNT_FILE = "./long-live.json"  # Update the path
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    try:
        token = generate_access_token(SERVICE_ACCOUNT_FILE, SCOPES)

        return jsonify({
            'message': "Token generated successfully.",
            'accessToken': token,
            'expiresAt': (datetime.now().timestamp() + 3600),
            'createdAt': datetime.now().isoformat() + 'Z'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error generating token: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
