from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class_labels = ["Bakso", "Gado-gado", "Gudeg", "Rendang", "Sate"]

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict-cnn', methods=['POST'])
def predict_cnn():
    model_path = "model-cnn.keras"
    try:
        print("Request received")
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['image']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        print(f"File saved at: {file_path}")

        model = load_model(model_path)
        print("Model loaded")

        img = load_img(file_path, target_size=(80, 120))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)

        predictions = model.predict(img_array)
        predicted_index = predictions.argmax()
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": float(confidence)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/predict-pretrained', methods=['POST'])
def predict_pretrained():
    model_path = "model-pretrained.keras"
    model = load_model(model_path)
    try:
        image_path = request.json.get("image")
        if not image_path:
            return jsonify({"error": "Image path is required"}), 400
        
        if not os.path.exists(image_path):
            return jsonify({"error": f"File not found: {image_path}"}), 404

        img = load_img(image_path, target_size=(80, 120))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)

        predictions = model.predict(img_array)
        predicted_index = predictions.argmax()
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)