from flask import Flask, request, jsonify
import base64
import os
from datetime import datetime

app = Flask(__name__)

# Configure upload folder inside container
UPLOAD_FOLDER = '/app/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    
def decode_base64(base64_string):
    """
    Decode base64 string to binary data
    """
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        raise Exception(f"Failed to decode base64: {str(e)}")

def generate_filename():
    """
    Generate unique filename using timestamp
    """
    return f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

def save_image(image_data, filename):
    """
    Save image data to file
    """
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        return file_path
    except Exception as e:
        raise Exception(f"Failed to save image: {str(e)}")

def process_base64_image(base64_string):
    """
    Process and save base64 image
    """
    try:
        # Decode base64
        image_data = decode_base64(base64_string)
        
        # Generate filename
        filename = generate_filename()
        
        # Save image
        save_image(image_data, filename)
        
        return filename
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/uploads', methods=['POST'])
def upload():
    try:
        # Get the JSON data
        data = request.get_json()
        
        # Validate request
        if 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image found in request'
            }), 400
        
        # Process image
        try:
            filename = process_base64_image(data['image'])
            return jsonify({
                'status': 'ok',
                'filename': filename
            }), 200
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Request processing error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))