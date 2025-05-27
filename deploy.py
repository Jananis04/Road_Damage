deploy .py
import cv2
import numpy as np
from keras.models import load_model

model = load_model(r'C:\Users\ADMIN\Desktop\NexGenMavericks\Czech\train_model.h5')

LABELS = ["DeepPothole", "Pothole", "Crack", "AlligatorCrack", "SlightDamage"]

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

def display_alerts(frame, predictions):
    for i, pred in enumerate(predictions):
        if pred > 0.5:
            label = LABELS[i]
            cv2.putText(frame, label, (50, 50 + 50*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

cap = cv2.VideoCapture(r'C:\Users\ADMIN\Desktop\NexGenMavericks\WhatsApp Video 2024-03-09 at 9.24.12 PM.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_frame = preprocess_frame(frame)

    predictions = model.predict(preprocessed_frame)[0]

    frame_with_alerts = display_alerts(frame, predictions)
    cv2.imshow('Road Damage Detection', frame_with_alerts)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
