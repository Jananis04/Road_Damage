test.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tkinter import Tk
from tkinter.filedialog import askopenfilename


model = load_model(r"C:\Users\ADMIN\Desktop\NexGenMavericks\Czech\train_model.h5")


LABELS = ["Crack", "Pothole", "Rut", "Spall"]
IMAGE_SIZE = (224, 224)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Resize image to the input size of the model
    image = cv2.resize(image, IMAGE_SIZE)
    # Convert image to array and preprocess for ResNet50 model
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

def draw_damage_boxes(image, predictions):
    for i, prediction in enumerate(predictions):
        predicted_label_index = np.argmax(prediction)
        if predicted_label_index < len(LABELS):
            predicted_label = LABELS[predicted_label_index]
            if predicted_label != "NoDamage":

                cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 2)
            else:
                cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
            cv2.putText(image, f"Damage: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

Tk().withdraw()
image_path = askopenfilename(title="Select an image file", filetypes=[("Image files", ".jpg;.jpeg;*.png")])


image = preprocess_image(image_path)

predictions = model.predict(np.expand_dims(image, axis=0))

processed_image = draw_damage_boxes(image.copy(), predictions)
cv2.imshow('Damage Detection', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
