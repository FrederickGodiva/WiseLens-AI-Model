from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class_labels = ["Bakso", "Gado-gado", "Gudeg", "Rendang", "Sate"]
nutrition_data = pd.read_csv("nutritions.csv")

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict-cnn', methods=['POST'])
def predict_cnn():
    try:
        print("Request received")
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['image']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        model = load_model("model-cnn.keras")

        img = load_img(file_path, target_size=(80, 120))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)

        predictions = model.predict(img_array)
        predicted_index = predictions.argmax()
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        print(predicted_label)

        nutrition = nutrition_data[nutrition_data['name'].str.lower().str.strip() == predicted_label.lower().strip()]
        if nutrition.empty:
            return jsonify({"error": f"Nutrition data not found for {predicted_label}"}), 404
        
        nutrition_dict = nutrition.iloc[0].to_dict()


        return jsonify({
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "nutritions": nutrition_dict
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/predict-pretrained', methods=['POST'])
def predict_pretrained():
    try:
        print("Request received")
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['image']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        model = load_model("model-pretrained.keras")

        img = load_img(file_path, target_size=(80, 120))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)

        predictions = model.predict(img_array)
        predicted_index = predictions.argmax()
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        print(predicted_label)

        nutrition = nutrition_data[nutrition_data['name'].str.lower().str.strip() == predicted_label.lower().strip()]
        if nutrition.empty:
            return jsonify({"error": f"Nutrition data not found for {predicted_label}"}), 404
        
        nutrition_dict = nutrition.iloc[0].to_dict()


        return jsonify({
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "nutritions": nutrition_dict
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)