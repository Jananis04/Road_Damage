
import os
import numpy as np
import tensorflow as tf
import keras

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import xml.etree.ElementTree as ET

LABELS = ["D00", "D10", "D20", "D40"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 50


def load_images_and_labels(image_paths, annotation_paths):
    images = []
    labels = []
    for img_path, ann_path in zip(image_paths, annotation_paths):
        image = load_img(img_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        images.append(image)
        annotation_labels = load_annotation_function(ann_path)
        labels.append(annotation_labels)

    return np.array(images), labels


def load_annotation_function(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    labels = []
    if root.find('object') is not None:
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(label)
    else:
        labels.append("NoDamage")

    return labels



region_train_data_dir = r"C:\Users\ADMIN\Desktop\NexGenMavericks\Czech\train"

image_dir = os.path.join(region_train_data_dir, "images")
annotation_dir = os.path.join(region_train_data_dir, "annotations", "xmls")

image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]
annotation_paths = [os.path.join(annotation_dir, ann) for ann in os.listdir(annotation_dir) if ann.endswith('.xml')]


train_images, train_labels = load_images_and_labels(image_paths, annotation_paths)


train_images = train_images / 255.0

train_labels = [[LABELS.index(label) if label in LABELS else len(LABELS) for label in labels] for labels in
                train_labels]
train_labels = [to_categorical(labels, num_classes=len(LABELS) + 1) for labels in train_labels]


train_labels_flat = [item for sublist in train_labels for item in sublist]

base_model = ResNet50(weights='imagenet', include_top=False)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(LABELS) + 1, activation='softmax')(x)  


model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)

# Train the model
model.fit(x=train_images, y=np.array(train_labels_flat), validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[checkpoint])

# Save trained model
model.save("trained_model.h5")
