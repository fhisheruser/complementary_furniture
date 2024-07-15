import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import os


# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
try:
    model = load_model('furniture_model.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Define the classes and mappings
classes = ['chair', 'table', 'sofa']
complementary_3d_models = {
    'chair': 'dataset/table/table1.jpeg',
    'table': 'dataset/chair/chair2.jpeg',
    'sofa': 'dataset/coffee_table/table4.jpeg'  # Example mappings
}

def prepare_image(image_path):
    try:
        img = load_img(image_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        logging.error(f"Error preparing image: {e}")
        return None

def predict_image(image_path):
    img_array = prepare_image(image_path)
    if img_array is not None:
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]
        return predicted_class
    else:
        return "Error processing image"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.debug("Received request")
        image = request.files['image']
        image_path = f"./{image.filename}"
        logging.debug(f"Saving image to {image_path}")
        image.save(image_path)
        
        # Ensure the image is saved before processing
        if os.path.exists(image_path):
            logging.debug(f"Image saved successfully: {image_path}")
            predicted_class = predict_image(image_path)
            logging.debug(f"Predicted class: {predicted_class}")

            result = complementary_3d_models.get(predicted_class, "No complementary furniture found")
            logging.debug(f"Result: {result}")
            return jsonify({'complementary_furniture_3d_model': result})
        else:
            logging.error("Failed to save image.")
            return jsonify({'error': "Failed to save image."}), 500

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# Serve static files (images and GLB models)
@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')