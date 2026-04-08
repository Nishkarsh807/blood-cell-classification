from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

model = load_model('model.h5')
classes = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = model.predict(img)[0]
    predicted_class = classes[np.argmax(pred)]

    confidence = round(np.max(pred) * 100, 2)

    return predicted_class, confidence, pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    result, confidence, probabilities = predict_image(filepath)

    return render_template('index.html',
                           prediction=result,
                           confidence=confidence,
                           probs=zip(classes, probabilities),
                           image=filepath)

if __name__ == "__main__":
    app.run(debug=True)