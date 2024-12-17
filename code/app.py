from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from bsk_ocr_1 import recognize_characters, preprocess_image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']
    
    
    image_data = image_data.split(',')[1]  
    image_bytes = base64.b64decode(image_data)
    
    # Convert bytes to NumPy array for OpenCV
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    # Preprocess and recognize characters using OCR
    character_images = preprocess_image(img)
    predictions = recognize_characters(character_images)

    # Check if '1' or '0' is recognized
    if 1 in predictions:
        return jsonify({'status': 'authentic'})  # Return status for authentic
    elif 0 in predictions:
        return jsonify({'status': 'fake'})  # Return status for fake
    else:
        return jsonify({'status': 'unknown'})  # In case no relevant characters are found

if __name__ == '__main__':
    app.run(debug=True)
