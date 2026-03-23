# esp_classify.py — runs on ESP32-S3 (MicroPython)
# Week 2 milestone: classifies a single pre-captured image stored as
# test_image.bin (32x32 raw uint8 bytes) using a CNN implemented in
# pure MicroPython (no TFLite / ulab required).
#
# Weights are loaded from binary float32 files in the /weights/ folder,
# exported from the trained Keras model by export_weights.py on the laptop.
#
# Network architecture (mirrors train_cnn.py):
#   Conv2D(1→8, 3x3, relu, same)  → MaxPool(2x2)
#   Conv2D(8→16, 3x3, relu, same) → MaxPool(2x2)
#   Conv2D(16→32, 3x3, relu, same)→ GlobalAvgPool
#   Dense(32→32, relu) → Dense(32→3, softmax)

import struct

CLASSES = ["rock", "paper", "scissors"]
IMG_SIZE = 32
W = "weights/"

# --- Load weights from binary files ---
# Each file contains flat float32 values matching Keras weight shapes.
def load_w(path):
    with open(path, 'rb') as f:
        data = f.read()
    return list(struct.unpack('{}f'.format(len(data) // 4), data))

print("Loading weights...")
c1_w = load_w(W + "0_conv2d_w.bin")       # shape (3,3,1,8)
c1_b = load_w(W + "0_conv2d_b.bin")       # shape (8,)
c2_w = load_w(W + "2_conv2d_1_w.bin")     # shape (3,3,8,16)
c2_b = load_w(W + "2_conv2d_1_b.bin")     # shape (16,)
c3_w = load_w(W + "4_conv2d_2_w.bin")     # shape (3,3,16,32)
c3_b = load_w(W + "4_conv2d_2_b.bin")     # shape (32,)
d1_w = load_w(W + "6_dense_w.bin")        # shape (32,32)
d1_b = load_w(W + "6_dense_b.bin")        # shape (32,)
d2_w = load_w(W + "8_dense_1_w.bin")      # shape (32,3)
d2_b = load_w(W + "8_dense_1_b.bin")      # shape (3,)
print("Weights loaded.")

# --- Forward pass helpers ---

def relu(x):
    return x if x > 0.0 else 0.0

def conv2d(inp, w, b, H, W_dim, C_in, C_out):
    """3x3 convolution with same-padding and ReLU activation.
    inp: flat list of shape H*W_dim*C_in (channels-last)
    w:   flat list matching Keras weight order kh*kw*C_in*C_out
    """
    out = [0.0] * (H * W_dim * C_out)
    for oh in range(H):
        for ow in range(W_dim):
            for oc in range(C_out):
                val = b[oc]
                for kh in range(3):
                    ih = oh + kh - 1
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(3):
                        iw = ow + kw - 1
                        if iw < 0 or iw >= W_dim:
                            continue
                        for ic in range(C_in):
                            val += (inp[ih * W_dim * C_in + iw * C_in + ic] *
                                    w[kh * 3 * C_in * C_out + kw * C_in * C_out + ic * C_out + oc])
                out[oh * W_dim * C_out + ow * C_out + oc] = relu(val)
    return out

def maxpool2d(inp, H, W_dim, C):
    """2x2 max-pooling with stride 2 — halves spatial dimensions."""
    oH, oW = H // 2, W_dim // 2
    out = [0.0] * (oH * oW * C)
    for oh in range(oH):
        for ow in range(oW):
            for c in range(C):
                ih, iw = oh * 2, ow * 2
                m = max(
                    inp[ih * W_dim * C + iw * C + c],
                    inp[ih * W_dim * C + (iw+1) * C + c],
                    inp[(ih+1) * W_dim * C + iw * C + c],
                    inp[(ih+1) * W_dim * C + (iw+1) * C + c]
                )
                out[oh * oW * C + ow * C + c] = m
    return out

def global_avg_pool(inp, H, W_dim, C):
    """Global average pooling: averages each channel over all spatial positions."""
    out = [0.0] * C
    n = H * W_dim
    for c in range(C):
        s = 0.0
        for i in range(n):
            s += inp[i * C + c]
        out[c] = s / n
    return out

def dense(inp, w, b, C_in, C_out, apply_relu=True):
    """Fully-connected layer with optional ReLU."""
    out = [0.0] * C_out
    for o in range(C_out):
        val = b[o]
        for i in range(C_in):
            val += inp[i] * w[i * C_out + o]
        out[o] = relu(val) if apply_relu else val
    return out

def softmax(x):
    """Numerically-stable softmax: subtract max before exponentiation."""
    m = max(x)
    e = [2.718281828 ** (v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

# --- Load test image ---
# test_image.bin is a 32x32 raw uint8 grayscale image (1024 bytes)
# created by prepare_test_image.py on the laptop.
print("Loading test image...")
with open("test_image.bin", "rb") as f:
    raw = f.read()

# Normalise pixel values from [0, 255] to [0.0, 1.0]
inp = [b / 255.0 for b in raw]
print("Running inference...")

# --- Forward pass ---
x = conv2d(inp, c1_w, c1_b, 32, 32, 1, 8)    # 32x32x1  -> 32x32x8
x = maxpool2d(x, 32, 32, 8)                    # 32x32x8  -> 16x16x8
x = conv2d(x, c2_w, c2_b, 16, 16, 8, 16)      # 16x16x8  -> 16x16x16
x = maxpool2d(x, 16, 16, 16)                   # 16x16x16 -> 8x8x16
x = conv2d(x, c3_w, c3_b, 8, 8, 16, 32)       # 8x8x16   -> 8x8x32
x = global_avg_pool(x, 8, 8, 32)               # 8x8x32   -> 32
x = dense(x, d1_w, d1_b, 32, 32, apply_relu=True)   # 32 -> 32
x = dense(x, d2_w, d2_b, 32, 3, apply_relu=False)   # 32 -> 3
probs = softmax(x)

# --- Result ---
idx = probs.index(max(probs))
print("Result: {} ({:.1f}%)".format(CLASSES[idx], probs[idx] * 100))
