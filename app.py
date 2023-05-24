from flask import Flask, render_template, request, redirect
import pymysql
from utils.img_preprocessing import img_resize, image_to_binary
import cv2

app = Flask(__name__)

db = pymysql.connect(host='localhost',
                    user='root',
                    password='0000',
                    db='testdb',
                    charset='utf8')

cursor = db.cursor()

@app.route('/')
def test():
    return render_template('index.html')

@app.route('/test')
def test_db():
    cursor.execute("select * from test;")
    result = list(cursor.fetchall())
    return result

@app.route('/upload_page', methods=['GET'])
def render_upload_page():
    return render_template('image_upload_test.html')

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    image2 = request.files['image2']  
    
    image.save('static/images/img_01.jpg')
    image2.save('static/images/img_02.jpg')

    image = img_resize('static/images/img_01.jpg')
    image2 = img_resize('static/images/img_02.jpg')

    cv2.imwrite('static/images/img_01.jpg', image)
    cv2.imwrite('static/images/img_02.jpg', image2)

    image = image_to_binary('static/images/img_01.jpg')
    image2 = image_to_binary('static/images/img_02.jpg')

    sql = "INSERT INTO testimg(img) VALUES (%s)"
    cursor.execute(sql, (image,))
    db.commit()
    return redirect('/')

if __name__ == '__main__': 
    app.run(debug=True) 


