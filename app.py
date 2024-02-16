import numpy as np
from flask import Flask, request, jsonify, render_template
import cv2
import requests
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
model = load_model("best_model.h5")

# Define a dictionary to map class numbers to class labels
ref = {
    0: "Alstonia Scholaris (P2)",
    1: "Arjun (P1)",
    2: "Bael (P4)",
    3: "Basil (P8)",
    4: "Chinar (P11)",
    5: "Guava (P3)",
    6: "Jamun (P5)",
    7: "Jatropha (P6)",
    8: "Lemon (P10)",
    9: "Mango (P0)",
    10: "Pomegranate (P9)",
    11: "Pongamia Pinnata (P7)",
}

@app.route("/predict/", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = request.json
    # Get the image URL or file path from the JSON data
    image_url = data.get("image_url")
    image_file = request.files.get("image_file")

    # Check if either image URL or image file is provided
    if not image_url and not image_file:
        return jsonify({"error": "No image URL or file provided"}), 400

    # If image URL is provided, download the image
    if image_url:
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image from provided URL"}), 400
        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    # If image file is provided, read the image from the file
    else:
        img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image
    img = cv2.resize(img, (256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    result = model.predict(img_array)

    # Assuming your model predicts a single class
    predicted_class = np.argmax(result)

    # Map the predicted class number to class label
    predicted_label = ref[predicted_class]

    return jsonify({"predicted_class": predicted_label})


if __name__ == "__main__":
    app.run(debug=True)
