from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

# Line Sweep Algorithm Function
def apply_line_sweep_algorithm(img):
    rows, cols = img.shape
    flagx = 0
    index_start_x = 0
    index_end_x = 0

    # Sweep along X-axis
    for i in range(rows):
        line = img[i, :]
        if flagx == 0:
            if 255 in line:
                index_start_x = i
                flagx = 1
        elif flagx == 1:
            if 255 in line:
                index_end_x = i
            elif index_start_x + 5 > index_end_x:
                index_start_x = 0
                flagx = 0
            else:
                break

    # Sweep along Y-axis
    flagy = 0
    index_start_y = 0
    index_end_y = 0

    for j in range(cols):
        line = img[index_start_x:index_end_x, j:j + 20]
        if flagy == 0:
            if 255 in line:
                index_start_y = j
                flagy = 1
        elif flagy == 1:
            if 255 in line:
                index_end_y = j
            elif index_start_y + 20 > index_end_y:
                index_start_y = 0
                flagy = 0
            else:
                break

    # Return an indication that the line sweep was successful
    return img[index_start_x:index_end_x, index_start_y:index_end_y]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image
    image_data = image_data.split(',')[1]  # Remove the data URL part
    image_bytes = base64.b64decode(image_data)

    # Convert bytes to NumPy array
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    # Apply Line Sweep Algorithm
    processed_image = apply_line_sweep_algorithm(img)

    # Simulating a result (can be replaced with your actual logic, e.g., using SVM or other methods)
    result_message = "Signature is authentic!" if np.random.random() > 0.5 else "Signature is forged or not valid!"

    return jsonify({'message': result_message})

if __name__ == '__main__':
    app.run(debug=True)
