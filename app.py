import argparse
from flask import Flask, render_template, request, redirect, send_from_directory
from utils.img_preprocessing import catch_face, is_image
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

test_model = load_model('models/model_siamese_neural_network.h5')

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
        return render_template('error.html', error='Error: you are not input file or file is not img', error_message="입력된 파일이 없거나 이미지가 아닙니다.")
    
    image = cv2.imread('static/images/img_01.jpg')
    image2 = cv2.imread('static/images/img_02.jpg')
    
    image = catch_face(image)
    image2 = catch_face(image2)

    if image is None or image2 is None:
        return render_template('error.html', error="Error: face detection failed", error_message="얼굴이 감지되지 않았습니다.")

    image = np.expand_dims(image, axis=0)
    image2 = np.expand_dims(image2, axis=0)  
    
    result = round(test_model.predict([image, image2])[0][0] * 100)

    return render_template('result.html', result=result, image_url='static/images/img_01.jpg', image2_url='static/images/img_02.jpg')

@app.route('/confetti_v2.js')
def serve_confetti():
    return send_from_directory('templates', 'confetti_v2.js')

@app.errorhandler(Exception)
def errorpage(error):
    error_table = {
        400: "잘못된 요청입니다.",
        401: "인증되지 않은 접근입니다.",
        403: "접근이 금지되었습니다.",
        404: "찾을 수 없는 페이지입니다. URL을 확인하고 다시 시도해주세요.",
        500: "내부 서버 오류가 발생했습니다."
    }

    error_code = getattr(error, 'code', 500)
    return render_template('error.html', error=error, error_message=error_table[error_code])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='Run the server in development mode')
    parser.add_argument('--product', action='store_true', help='Run the server in production mode')
    args = parser.parse_args()

    if args.dev:
        app.run(host="127.0.0.1", debug=True, port="5000")
    elif args.product:
        app.run(host="0.0.0.0", debug=False, port="80")
    else:
        parser.print_help()

