# prepare_test_image.py — runs on the laptop
# Converts a dataset PNG image to a raw 32x32 uint8 binary file
# (test_image.bin) that can be uploaded to the ESP32 and classified
# by esp_classify.py without any image-decoding library on the device.
#
# Usage:
#   python prepare_test_image.py dataset/rock/rock_0000.png
# Defaults to rock_0000.png if no argument is given.

import struct
from PIL import Image
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "dataset/rock/rock_0000.png"

# Open image, convert to grayscale, resize to 32x32
img = Image.open(path).convert("L").resize((32, 32))
pixels = list(img.getdata())  # 1024 uint8 values, row-major

with open("test_image.bin", "wb") as f:
    f.write(bytes(pixels))

print(f"Saved test_image.bin  ({len(pixels)} bytes)  from {path}")
