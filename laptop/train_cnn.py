# train_cnn.py — runs on the laptop
# Trains a small CNN on the rock/paper/scissors dataset collected via
# laptop_client.py and saves the model to rps_model.keras.
#
# Architecture chosen to run inference in pure MicroPython on the ESP32:
#   - GlobalAveragePooling replaces Flatten+Dense, keeping weight count small
#   - Three conv blocks with MaxPooling: 32x32 -> 16x16 -> 8x8
#   - Final dense layers: 64 -> 64 -> 3 (softmax)
#
# Input: 32x32 grayscale images (matching ESP32 preprocessing)
# Output: probability distribution over [rock, paper, scissors]

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

DATASET_DIR = "dataset"
CLASSES = ["rock", "paper", "scissors"]
IMG_SIZE = 32

# --- Load dataset ---
images = []
labels = []

for label_idx, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_DIR, cls)
    for fname in os.listdir(folder):
        if not fname.endswith(".png"):
            continue
        path = os.path.join(folder, fname)
        img = tf.keras.utils.load_img(path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.utils.img_to_array(img) / 255.0  # normalise to [0, 1]
        images.append(arr)
        labels.append(label_idx)

images = np.array(images)
labels = np.array(labels)

print(f"Loaded {len(images)} images — rock:{np.sum(labels==0)}  paper:{np.sum(labels==1)}  scissors:{np.sum(labels==2)}")

# 80/20 train/validation split using a random permutation
indices = np.random.permutation(len(images))
split = int(0.8 * len(images))
train_idx, val_idx = indices[:split], indices[split:]

x_train, y_train = images[train_idx], labels[train_idx]
x_val,   y_val   = images[val_idx],   labels[val_idx]

# --- CNN architecture ---
# GlobalAveragePooling2D avoids a large Flatten layer so the total
# parameter count stays small enough to fit in ESP32 RAM.
model = keras.Sequential([
    keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),  # -> 32x32x16
    keras.layers.MaxPooling2D((2, 2)),                                     # -> 16x16x16

    keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),  # -> 16x16x32
    keras.layers.MaxPooling2D((2, 2)),                                     # -> 8x8x32

    keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),  # -> 8x8x64
    keras.layers.GlobalAveragePooling2D(),                                 # -> 64

    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.4),   # regularisation to reduce over-fitting
    keras.layers.Dense(len(CLASSES), activation="softmax"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Data augmentation: small rotations, shifts, and horizontal flips
# increase variety without collecting extra images.
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

model.fit(
    datagen.flow(x_train, y_train, batch_size=16),
    epochs=80,
    validation_data=(x_val, y_val),
)

model.save("rps_model.keras")
print("Model saved to rps_model.keras")

loss, acc = model.evaluate(x_val, y_val, verbose=0)
print(f"Validation accuracy: {acc*100:.1f}%")
