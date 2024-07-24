import pyvjoy
import time
import pygame

time.sleep(1)
print("1")
time.sleep(1)
print("2")
time.sleep(1)
print("3")
vj = pyvjoy.VJoyDevice(1)
vj.set_axis(pyvjoy.HID_USAGE_X, 0x0000)
time.sleep(1)
print("2")
vj.set_axis(pyvjoy.HID_USAGE_X, 0x8000)
time.sleep(1)
print("3")
vj.set_axis(pyvjoy.HID_USAGE_X, 0x4000)

vj.set_axis(pyvjoy.HID_USAGE_RZ, 0x8000)

time.sleep(1)