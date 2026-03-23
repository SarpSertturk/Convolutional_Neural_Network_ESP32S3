# streaming_server.py — runs on ESP32-S3 (MicroPython)
# Connects to Wi-Fi, initialises the camera in GRAYSCALE mode,
# and streams raw frames over TCP to a laptop client.
# Each frame is prefixed with a 4-byte big-endian length header.

import network
import time
import socket as soc
import esp
import struct
from camera import Camera, PixelFormat, FrameSize

# Suppress debug output from the ESP SDK
esp.osdebug(None)

# --- Wi-Fi setup ---
wlan = network.WLAN(network.STA_IF)
wlan.active(False)
time.sleep(1)
wlan.active(True)

WIFI_SSID = 'Vilo_93a7'
WIFI_PASS = 'J4nxzDtZ'
wlan.connect(WIFI_SSID, WIFI_PASS)

# Wait up to 60 seconds for a connection
timeout = 60
while not wlan.isconnected() and timeout > 0:
    print("Waiting for WiFi... ({} sec remaining)".format(timeout))
    time.sleep(5)
    timeout -= 5

if wlan.isconnected():
    ip = wlan.ifconfig()[0]
    print("WiFi connected!")
    print("ESP32 IP address:", ip)
    print(">>> Set HOST =", repr(ip), "in laptop_client.py")
else:
    print("WiFi failed. Check SSID/password and that your hotspot is on.")
    raise SystemExit

# --- Camera initialisation ---
# GRAYSCALE gives 160x120 frames (19 200 bytes each), one byte per pixel.
# This format avoids JPEG decoding overhead and gives exact pixel values.
cam = Camera(pixel_format=PixelFormat.GRAYSCALE)
cam.init()
print("Camera initialised")

# --- TCP server: listen on port 9999 ---
PORT = 9999
addr = soc.getaddrinfo('0.0.0.0', PORT)[0][-1]
s = soc.socket(soc.AF_INET, soc.SOCK_STREAM)
s.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR, 1)
s.bind(addr)
s.listen(1)
print("Server ready — waiting for client on port", PORT)
print("Run laptop_client.py on your laptop now.")

# --- Main loop: accept clients and stream frames ---
while True:
    cs, ca = s.accept()
    print("Client connected from:", ca)
    frame_count = 0
    try:
        while True:
            img = cam.capture()
            if img:
                size = len(img)
                # Send 4-byte big-endian length header then raw pixel data
                cs.write(struct.pack('>I', size))
                cs.write(img)
                frame_count += 1
                if frame_count % 30 == 0:
                    print("Frames sent:", frame_count)
    except Exception as e:
        print("Client disconnected after {} frames: {}".format(frame_count, e))
        cs.close()
    # Loop back to s.accept() to wait for the next client
