# export_weights.py — runs on the laptop
# Exports all trainable weights from rps_model.keras as raw binary
# float32 files so they can be uploaded to the ESP32 and loaded by
# esp_classify.py / esp_realtime.py without any ML framework.
#
# Output files are written to the 'weights/' folder.
# Naming convention: {layer_index}_{layer_name}_{w|b}.bin
#   w = kernel (weight matrix)
#   b = bias vector

import numpy as np
import tensorflow as tf
import struct
import os

MODEL_PATH = "rps_model.keras"
OUTPUT_DIR = "weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# Iterate over every layer; skip layers with no trainable parameters
# (e.g. MaxPooling, Dropout, GlobalAveragePooling).
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if not weights:
        continue
    name = layer.name
    for j, w in enumerate(weights):
        tag = "w" if j == 0 else "b"
        filename = os.path.join(OUTPUT_DIR, f"{i}_{name}_{tag}.bin")
        # Flatten to 1-D and cast to float32 (Keras default is float32)
        flat = w.flatten().astype(np.float32)
        with open(filename, "wb") as f:
            f.write(struct.pack(f"{len(flat)}f", *flat))
        print(f"Saved {filename}  shape={w.shape}  bytes={os.path.getsize(filename)}")

print("\nDone. Upload the 'weights' folder to ESP32.")
