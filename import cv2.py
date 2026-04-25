import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

classes = ['10','20','50','100','200','500','2000']

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame.copy()

    img = preprocess(roi)
    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    label = f"{classes[class_index]} Rs ({confidence:.2f})"

    cv2.putText(frame, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Currency Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()