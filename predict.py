import cv2
from tensorflow.keras.models import load_model
from utils import preprocess_image, get_label

model = load_model("model.h5")

img = cv2.imread("test.jpg")  # put your image here

processed = preprocess_image(img)
pred = model.predict(processed)

label, conf = get_label(pred)

print(f"Prediction: ₹{label} ({conf:.2f})")

cv2.putText(img, f"{label} ({conf:.2f})",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,255,0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()