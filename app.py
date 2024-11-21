from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2

from utils.util import three_eye_detect, is_blurry_image, detection_tflite, process_dots
from utils.model_loader import EYE_DETECTION, MAIN_DETECTION, DOT_DETECTION

app = Flask(__name__)


# Set upload folder for temporary storage
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({"error": "No image file selected."}), 400

    try:
        # Save the uploaded image temporarily
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        image = cv2.imread(image_path)

        eye_detected = three_eye_detect(image, EYE_DETECTION)
        if eye_detected:
            # is_blur = is_blurry_image(image)
            # if is_blur:
            #     rtrn = 4
            # else:
                main_result = detection_tflite(image, MAIN_DETECTION)
                if len(main_result) >= 1:
                    dot_detect = process_dots(image, DOT_DETECTION)
                    if dot_detect:
                        rtrn = 1
                    else:
                        rtrn = 2
                else:
                    rtrn = 2

        else:
            rtrn = 3

        return jsonify({
            "returned": rtrn

                        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
