import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('model.h5')
classes = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))
    pred = model.predict(img)
    return classes[np.argmax(pred)]
