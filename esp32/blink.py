# blink.py — runs on ESP32-S3 (MicroPython)
# Blinks the onboard LED to verify the board is flashed and running correctly.
# This is the first program run after flashing the MicroPython firmware.

import machine
import time

# GPIO 48 is the onboard RGB LED on most ESP32-S3 boards
led = machine.Pin(48, machine.Pin.OUT)

print("Blink started")

while True:
    led.on()
    time.sleep(0.5)
    led.off()
    time.sleep(0.5)
