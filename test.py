import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import xml.etree.ElementTree as ET

# Constants
LABELS = ["D00", "D10", "D20", "D40"]
IMAGE_SIZE = (224, 224)
TEST_IMAGE_DIR = "/content/drive/MyDrive/TriNIT/dataset/Czech/test/images"
OUTPUT_FILE = "test_results.txt"


def load_annotation_function(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Check if the image has any road damage
    has_damage = False
    if root.find('object') is not None:
        has_damage = True

    return has_damage


def load_and_predict_images(model, image_dir):
    results = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(image_dir, img_file)
            image = load_img(img_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            image = image / 255.0  # Normalize

            # Predict classes
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            if predicted_class < len(LABELS):
                predicted_label = LABELS[predicted_class]
            else:
                predicted_label = "NoDamage"

            # Load and process annotations to check for road damage
            annotation_path = os.path.join(TEST_IMAGE_DIR, img_file.replace(".jpg", ".xml"))
            has_damage = load_annotation_function(annotation_path)

            # Append results
            results.append((img_file, predicted_label, has_damage))
    return results


def save_results(results, output_file):
    with open(output_file, "w") as f:
        for img_file, predicted_label, has_damage in results:
            f.write(f"Image: {img_file}, Predicted Label: {predicted_label}, Has Damage: {has_damage}\n")


# Load trained model
model = load_model("trained_model.h5")

# Load and predict images
results = load_and_predict_images(model, TEST_IMAGE_DIR)

# Save results to a text file
save_results(results, OUTPUT_FILE)