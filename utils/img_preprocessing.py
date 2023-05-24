import cv2
import os

def img_resize(first_img):
    img = cv2.imread(first_img)
    resized_image = cv2.resize(img, (105, 105))
    face_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return face_image

def image_to_binary(image_path):
    with open(image_path, 'rb') as f:
        binary_data = f.read()
    return binary_data