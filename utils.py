import cv2
import numpy as np

IMG_SIZE = 224

classes = ['10','20','50','100','200','500','2000']

def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    return img

def get_label(pred):
    class_index = np.argmax(pred)
    confidence = np.max(pred)
    return classes[class_index], confidence