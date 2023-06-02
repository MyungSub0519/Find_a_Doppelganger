from flask import Flask, render_template, request, redirect
from utils.img_preprocessing import catch_face
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

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

if __name__ == '__main__': 
    app.run(debug=True)