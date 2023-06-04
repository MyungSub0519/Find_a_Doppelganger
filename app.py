from flask import Flask, render_template, request, redirect, send_from_directory
from utils.img_preprocessing import catch_face, is_image
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import re
import os

test_model = load_model('models\model_siamese_neural_network.h5')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def render_upload_page():
    return render_template('index.html')        

@app.route('/upload', methods=['POST'])
def upload():   
    image = request.files['image']
    image2 = request.files['image2'] 

    image.save('static/images/img_01.jpg')
    image2.save('static/images/img_02.jpg')

    if is_image(image,image2):
        return 'Error: Input file is not img', 400
    
    image = cv2.imread('static/images/img_01.jpg')
    image2 = cv2.imread('static/images/img_02.jpg')
    
    image = catch_face(image)
    image2 = catch_face(image2)

    if image is None or image2 is None:
        return 'Error: face detection failed', 400

    image = np.expand_dims(image, axis=0)
    image2 = np.expand_dims(image2, axis=0)  
    
    result = round(test_model.predict([image, image2])[0][0] * 100)

    return render_template('result.html', result=result, image_url='static/images/img_01.jpg', image2_url='static/images/img_02.jpg')



@app.route('/confetti_v2.js')
def serve_confetti():
    return send_from_directory('templates', 'confetti_v2.js')

if __name__ == '__main__': 
    app.run(host="0.0.0.0",  debug=True)
