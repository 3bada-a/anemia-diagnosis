import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode (avoids CUDA toolkit issues in WSL)

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_anemia.h5')
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Dynamically detect input shape from model
input_shape = model.input_shape  # e.g. (None, 224, 224, 3)
IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]

print(f"Model loaded. Expected input shape: {input_shape}")
print(f"Will resize images to: {IMG_WIDTH}x{IMG_HEIGHT}")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        # Binary classification: low value = Anemic, high value = Not Anemic
        is_anemic = confidence < 0.5
        result = {
            'prediction': 'Anemic' if is_anemic else 'Not Anemic',
            'confidence': (1 - confidence) if is_anemic else confidence
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_input_shape': str(input_shape)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
