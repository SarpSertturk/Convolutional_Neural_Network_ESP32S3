# laptop_client.py — runs on the laptop
# Connects to the ESP32 TCP streaming server, displays the live camera
# feed, runs the trained Keras model for live classification, and lets
# the user press R / P / S to save labelled training images.
#
# Saved images use the same nearest-neighbour sampling as the ESP32
# so that training data and inference conditions match exactly.

import socket
import struct
import os
import cv2
import numpy as np
import tensorflow as tf

HOST = "192.168.58.210"   # ESP32 IP address (update if it changes)
PORT = 9999

DATASET_DIR = "dataset"
CLASSES = ["rock", "paper", "scissors"]
KEYS = {ord("r"): "rock", ord("p"): "paper", ord("s"): "scissors"}

IMG_SIZE = 32
MODEL_PATH = "rps_model.keras"

def setup_dataset_dirs():
    """Create dataset/rock, dataset/paper, dataset/scissors if absent."""
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

def count_images():
    """Return a dict with the current image count per class."""
    return {cls: len(os.listdir(os.path.join(DATASET_DIR, cls))) for cls in CLASSES}

def save_frame(frame, label, counts):
    """Save one labelled image using the same nearest-neighbour
    downsampling that the ESP32 uses during inference."""
    H, W = frame.shape[:2]  # 120 x 160
    out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for dy in range(IMG_SIZE):
        sy = int(dy * H / IMG_SIZE)
        for dx in range(IMG_SIZE):
            sx = int(dx * W / IMG_SIZE)
            out[dy, dx] = frame[sy, sx, 0] & 0xFF  # force unsigned
    idx = counts[label]
    path = os.path.join(DATASET_DIR, label, f"{label}_{idx:04d}.png")
    cv2.imwrite(path, out)
    counts[label] += 1
    print(f"Saved {path}  |  rock:{counts['rock']}  paper:{counts['paper']}  scissors:{counts['scissors']}")

def predict(model, frame):
    """Run the Keras model on one BGR frame and return (class, confidence)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    inp = resized.astype("float32") / 255.0
    inp = inp.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

def main():
    setup_dataset_dirs()
    counts = count_images()

    # Load model if available; fall back to capture-only mode
    model = None
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH} — live classification ON")
    else:
        print(f"No model found at {MODEL_PATH} — capture mode only")

    print(f"Connecting to ESP32 at {HOST}:{PORT} ...")
    print("Press R=rock  P=paper  S=scissors  Q=quit")
    print(f"Existing images — rock:{counts['rock']}  paper:{counts['paper']}  scissors:{counts['scissors']}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected!")

        frame_count = 0
        last_label, last_conf = "", 0.0
        while True:
            frame_count += 1

            # Read 4-byte big-endian length header sent by the ESP32
            header = b""
            while len(header) < 4:
                chunk = s.recv(4 - len(header))
                if not chunk:
                    print("Connection closed.")
                    return
                header += chunk
            size = struct.unpack(">I", header)[0]

            # Read the raw grayscale frame (160x120 = 19 200 bytes)
            data = b""
            while len(data) < size:
                chunk = s.recv(size - len(data))
                if not chunk:
                    print("Connection closed.")
                    return
                data += chunk

            # Reshape to (120, 160) and convert to BGR for OpenCV display
            gray = np.frombuffer(data, dtype=np.uint8).reshape(120, 160)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Run model every 5 frames to keep the display responsive
            if model is not None:
                if frame_count % 5 == 0:
                    last_label, last_conf = predict(model, frame)
                text = f"{last_label}  {last_conf*100:.0f}%"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("ESP32 Camera  |  R=rock  P=paper  S=scissors  Q=quit", frame)
            cv2.moveWindow("ESP32 Camera  |  R=rock  P=paper  S=scissors  Q=quit", 100, 100)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in KEYS:
                save_frame(frame, KEYS[key], counts)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
