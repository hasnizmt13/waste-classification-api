from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# Load the model
MODEL_PATH = 'waste_model.h5'
model = load_model(MODEL_PATH, compile=False)

def image_processing(img):
    '''Function for image processing'''
    img_resized = img.resize((64, 64))
    img_array = img_to_array(img_resized)
    img_array = img_array.reshape((1, 64, 64, 3))
    return img_array

def predict_waste(image_array):
    '''Function to predict waste type'''
    result = model.predict(image_array)
    prediction = 'Recyclable Waste' if result[0][0] == 1 else 'Organic Waste'
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img_array = image_processing(img)
        prediction = predict_waste(img_array)
        return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
