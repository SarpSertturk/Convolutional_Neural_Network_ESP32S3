# esp_realtime.py — runs on ESP32-S3 (MicroPython)
# Week 3 milestone: captures live frames from the camera and classifies
# each one as rock, paper, or scissors using the CNN inference engine
# implemented in pure MicroPython.
#
# After each inference the camera needs time to re-stabilise its
# auto-exposure (inference takes ~30 s), so 60 warm-up frames are
# captured and discarded before each real capture.
#
# Network architecture (mirrors train_cnn.py):
#   Conv2D(1→16, 3x3, relu, same) → MaxPool(2x2)
#   Conv2D(16→32, 3x3, relu, same)→ MaxPool(2x2)
#   Conv2D(32→64, 3x3, relu, same)→ GlobalAvgPool
#   Dense(64→64, relu) → Dense(64→3, softmax)

import struct
import time
from camera import Camera, PixelFormat, FrameSize

CLASSES = ["rock", "paper", "scissors"]
W = "weights/"

# --- Load weights ---
# Binary float32 files exported from Keras by export_weights.py
def load_w(path):
    with open(path, 'rb') as f:
        data = f.read()
    return list(struct.unpack('{}f'.format(len(data) // 4), data))

print("Loading weights...")
c1_w = load_w(W + "0_conv2d_w.bin")
c1_b = load_w(W + "0_conv2d_b.bin")
c2_w = load_w(W + "2_conv2d_1_w.bin")
c2_b = load_w(W + "2_conv2d_1_b.bin")
c3_w = load_w(W + "4_conv2d_2_w.bin")
c3_b = load_w(W + "4_conv2d_2_b.bin")
d1_w = load_w(W + "6_dense_w.bin")
d1_b = load_w(W + "6_dense_b.bin")
d2_w = load_w(W + "8_dense_1_w.bin")
d2_b = load_w(W + "8_dense_1_b.bin")
print("Weights loaded.")

# --- Forward pass ---

def relu(x):
    return x if x > 0.0 else 0.0

def conv2d(inp, w, b, H, W_dim, C_in, C_out):
    """3x3 convolution with same-padding and ReLU (channels-last layout)."""
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
    """2x2 max-pooling with stride 2."""
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
    """Reduces each channel to a single value by spatial averaging."""
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
    """Numerically-stable softmax."""
    m = max(x)
    e = [2.718281828 ** (v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

def classify(inp):
    """Run the full CNN forward pass and return (class_name, confidence)."""
    x = conv2d(inp, c1_w, c1_b, 32, 32, 1, 16)
    print("c1 mean:{:.3f} max:{:.3f}".format(sum(x)/len(x), max(x)))
    x = maxpool2d(x, 32, 32, 16)
    x = conv2d(x, c2_w, c2_b, 16, 16, 16, 32)
    print("c2 mean:{:.3f} max:{:.3f}".format(sum(x)/len(x), max(x)))
    x = maxpool2d(x, 16, 16, 32)
    x = conv2d(x, c3_w, c3_b, 8, 8, 32, 64)
    print("c3 mean:{:.3f} max:{:.3f}".format(sum(x)/len(x), max(x)))
    x = global_avg_pool(x, 8, 8, 64)
    print("gap:", [round(v,3) for v in x[:8]])
    x = dense(x, d1_w, d1_b, 64, 64, apply_relu=True)
    x = dense(x, d2_w, d2_b, 64, 3, apply_relu=False)
    print("logits:", [round(v,3) for v in x])
    probs = softmax(x)
    idx = probs.index(max(probs))
    return CLASSES[idx], probs[idx]

def preprocess(frame, src_w=160, src_h=120, dst=32):
    """Downsample a 160x120 grayscale frame to 32x32 using nearest-neighbour
    sampling, then normalise to [0.0, 1.0].
    The & 0xFF mask ensures correct unsigned values on MicroPython."""
    inp = []
    for dy in range(dst):
        sy = int(dy * src_h / dst)
        for dx in range(dst):
            sx = int(dx * src_w / dst)
            inp.append((frame[sy * src_w + sx] & 0xFF) / 255.0)
    return inp

# --- Camera initialisation ---
print("Starting camera...")
cam = Camera(pixel_format=PixelFormat.GRAYSCALE)
cam.init()
print("Camera ready. Classifying...")

# --- Main loop ---
while True:
    # Flush stale frames so auto-exposure can stabilise after the long
    # inference pause (~30 s).  60 frames x 100 ms = 6 s warm-up.
    for _ in range(60):
        cam.capture()
        time.sleep_ms(100)

    frame = cam.capture()
    if frame:
        inp = preprocess(frame)
        print("inp mean:{:.3f} first:{:.3f}".format(sum(inp)/len(inp), inp[500]))
        label, conf = classify(inp)
        print(">> {} ({:.1f}%)".format(label, conf * 100))
