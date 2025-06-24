# app.py

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("Blood Cell.h5")

classes = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    img_path = os.path.join('static', img.filename)
    img.save(img_path)

    img_loaded = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_loaded)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    result = classes[np.argmax(pred)]

    return render_template('result.html', prediction=result, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
