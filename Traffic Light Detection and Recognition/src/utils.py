import cv2
import numpy as np

def get_signal_color(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    yellow_mask = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))
    green_mask  = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

    r, y, g = cv2.countNonZero(red_mask), cv2.countNonZero(yellow_mask), cv2.countNonZero(green_mask)

    if max(r, y, g) == 0:
        return "unknown"
    if r > y and r > g:
        return "red"
    elif y > r and y > g:
        return "yellow"
    else:
        return "green"
