# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import urllib.request
import base64
import os
import json
from cellsize import classify_cell_size
from nucleussize import process_nucleus_image
from hyperchromasia import detect_hyperchromasia
import requests  # Add this import for making HTTP requests
from increasednucleoli import detect_nucleoli
from increasednucleoliTest import modelOutput
from mitosis import detect_mitotic_figures
from keratin import detect_keratin_figures
from ncratio import process_ratio
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from werkzeug.utils import secure_filename
from features import classify_cell_size, process_nucleus_image, detect_hyperchromasia, detect_keratin_figures, detect_mitotic_figures


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


def model_output(image_path):
    # Load the model
    model = load_model("model/keras_Model.h5", compile=False)

    # Load the labels
    with open("model/labels.txt", "r") as file:
        class_names = [line.strip() for line in file.readlines()]

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # Print prediction and confidence score
    print("Class:", class_name)
    print("Confidence Score:", confidence_score)

    return {
        "class": class_name,
        "accuracy": confidence_score
    }

@app.route('/api/model', methods=['POST'])
def model():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Java file not provided'})
        
        image_file = request.files.get('image')

        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        print(file_path)
        result = model_output(file_path)

        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/api/cell_size', methods=['POST'])
def cell_size_route():
    try:
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        dataset_path = './dataset'  # Provide the actual dataset path

        # Read image file
        image_bytes = image_file.read()

        # Call the classification function with the dataset path
        result = classify_cell_size(image_bytes, dataset_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/nucleus_size', methods=['POST'])
def nucleus_size():
    try:
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()

        # Call the nucleus size processing function
        result = process_nucleus_image(image_bytes)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/hyperchromasia', methods=['POST'])
def hyperchromasia_route():
    try:
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()

        # Call the hyperchromasia detection function
        result = detect_hyperchromasia(image_bytes)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/increasednucleoli-test', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file part in the request', 400
    
    file = request.files.get('image')
    
    if file.filename == '':
        return 'No selected file', 400
    
    file.save('uploads/' + file.filename)
    result = modelOutput('uploads/' + file.filename)
    return jsonify(result)

@app.route('/api/increasednucleoli', methods=['POST'])  
def increased_nucleoli_route():
    try:
        # Read the image data from the request
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()
        # Extract the JSON response from the API
        result = detect_nucleoli(image_bytes)
        
        # Return the result obtained from the increased_nucleoli API
        return jsonify(result)

    except Exception as e:
        # Return error response
        print(e)
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/mitosis', methods=['POST'])  
def mitosis():
    try:
        # Read the image data from the request
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()
        template_folder_path = "./mitosistemp"
        # Extract the JSON response from the API
        result = detect_mitotic_figures(image_bytes, template_folder_path)
        
        # Return the result obtained from the increased_nucleoli API
        return jsonify(result)

    except Exception as e:
        # Return error response
        print(e)
        return jsonify({'error': str(e)}), 500    
    

@app.route('/api/keratin', methods=['POST'])  
def keratin():
    try:
        # Read the image data from the request
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()
        template_folder_path = "./keratintemp"
        # Extract the JSON response from the API
        result = detect_keratin_figures(image_bytes, template_folder_path)
        
        # Return the result obtained from the increased_nucleoli API
        return jsonify(result)

    except Exception as e:
        # Return error response
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/ncratio', methods=['POST'])  
def ncratio():
    try:
        # Read the image data from the request
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})

        # Read image file
        image_bytes = image_file.read()
        # Extract the JSON response from the API
        result = process_ratio(image_bytes)
        
        # Return the result obtained from the increased_nucleoli API
        return jsonify(result)

    except Exception as e:
        # Return error response
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/features', methods=['POST'])  
def features():
    try:
        # Read the image data from the request
        image_file = request.files.get('image')
        if image_file is None:
            return jsonify({'error': 'No image file provided'})
        dataset_path = './dataset'
        # Read image file
        image_bytes = image_file.read()
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        print(file_path)
        template_folder_path = "./mitosistemp"
        template_folder_path_k = "./keratintemp"

        # Extract the JSON response from the API
        resultCell = classify_cell_size(image_bytes, dataset_path)
        resultNucleus = process_nucleus_image(image_bytes)
        resultHyper = detect_hyperchromasia(image_bytes)
        resultMitosis = detect_mitotic_figures(image_bytes, template_folder_path)
        resultKeratin = detect_keratin_figures(image_bytes, template_folder_path_k)

        print(resultCell['totalCellSize'])

        # Return the result obtained from the increased_nucleoli API
        return jsonify(resultCell, resultNucleus, resultHyper, resultMitosis, resultKeratin)
    

    except Exception as e:
        # Return error response
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
