import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import time

model = load_model(r"mobilenet/model.h5", compile=False)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def get_class_label(class_index):
    class_labels = {0: 'No Fire', 1: 'Fire!'}
    return class_labels.get(class_index, "Unknown")


try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        resized_frame = cv2.resize(frame, (160, 160))

        tensor = resized_frame.reshape(-1, 160, 160, 3)
        pred = model.predict(tensor)[0][0]
        class_label = get_class_label(1 if pred > 0 else 0)

        cv2.putText(frame, f'Prediction: {class_label}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam.release()
    cv2.destroyAllWindows()
