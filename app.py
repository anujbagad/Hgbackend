# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import urllib.request
import base64
import json
from cellsize import classify_cell_size
from nucleussize import process_nucleus_image
from hyperchromasia import detect_hyperchromasia
import requests  # Add this import for making HTTP requests
from increasednucleoli import increased_nuclei

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
CORS(app)

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
        result = increased_nuclei(image_bytes)
        
        # Return the result obtained from the increased_nucleoli API
        return jsonify(result)

    except Exception as e:
        # Return error response
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
