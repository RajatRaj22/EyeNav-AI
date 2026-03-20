# utils/drawing.py

import cv2

def put_text(frame, text, pos, color=(0,255,0), scale=0.7):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
