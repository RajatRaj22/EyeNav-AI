"""
main_gui.py
Hybrid Eye-Control GUI: gaze navigation + blink-based Morse typing with predictive suggestions.

Run:
    python main_gui.py
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from collections import defaultdict
from difflib import get_close_matches
import threading

# ---------------- Settings (tweakable) ----------------
CAM_INDEX = 0
SMOOTH_LEN = 5
HORIZ_THRESHOLD = 0.18
VERT_THRESHOLD = 0.18
ACTION_DELAY = 0.7
DOT_DASH_SEP = 0.45   # seconds: < => dot, >= => dash
WORD_PAUSE = 1.2      # seconds to commit morse char/word
MIN_BLINK_IGNORE = 0.06

pyautogui.FAILSAFE = True

# ---------------- Morse dictionary ----------------
MORSE_DICT = {
    '.-':'A','-...':'B','-.-.':'C','-..':'D','.':'E','..-.':'F',
    '--.':'G','....':'H','..':'I','.---':'J','-.-':'K','.-..':'L',
    '--':'M','-.':'N','---':'O','.--.':'P','--.-':'Q','.-.':'R',
    '...':'S','-':'T','..-':'U','...-':'V','.--':'W','-..-':'X',
    '-.--':'Y','--..':'Z',
    '-----':'0','.----':'1','..---':'2','...--':'3','....-':'4',
    '.....':'5','-....':'6','--...':'7','---..':'8','----.':'9'
}

# ---------------- Predictive bigram (small corpus) ----------------
CORPUS = "hello world this is an example of predictive text for morse input system demo hello world test typing"
words = CORPUS.split()
bigrams = defaultdict(list)
for i in range(len(words)-1):
    bigrams[words[i]].append(words[i+1])
DICTIONARY = list(set(words))

def predict_next(word):
    return bigrams.get(word.lower(), [])[:3]

def correct_word(word):
    matches = get_close_matches(word.lower(), DICTIONARY, n=1, cutoff=0.7)
    return matches[0] if matches else word

# ---------------- MediaPipe setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Landmarks indices (MediaPipe)
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_TOP = 159
LEFT_BOTTOM = 145
RIGHT_TOP = 386
RIGHT_BOTTOM = 374

# ---------------- App State ----------------
center_h, center_v = 0.5, 0.5
h_queue = []
v_queue = []
calibrated = False
last_action_time = 0.0

blink_active = False
blink_start = 0.0
last_blink_time = 0.0
morse_buffer = ''
typed_text = ''
char_commit_time = 0.0

open_history = []

# Thread-safe frame
frame_lock = threading.Lock()
latest_frame = None
running = True

# ---------------- Helper functions ----------------
def mean_point(pts):
    pts = np.array(pts)
    return np.mean(pts, axis=0).astype(int)

def decode_morse(seq):
    return MORSE_DICT.get(seq, '?')

# ---------------- Camera Thread ----------------
def camera_loop():
    global latest_frame, center_h, center_v, h_queue, v_queue
    global calibrated, last_action_time
    global blink_active, blink_start, last_blink_time, morse_buffer, typed_text, char_commit_time
    global open_history

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = face_mesh.process(frame_rgb)
        gaze = "CENTER"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            mesh = np.array([(int(p.x*w), int(p.y*h)) for p in lm])

            # iris centers
            l_iris = mean_point([mesh[i] for i in LEFT_IRIS])
            r_iris = mean_point([mesh[i] for i in RIGHT_IRIS])
            cv2.circle(frame, tuple(l_iris), 3, (0,255,0), -1)
            cv2.circle(frame, tuple(r_iris), 3, (0,255,0), -1)

            # horizontal normalized per-eye
            llx = mesh[LEFT_EYE_LEFT][0]; lrx = mesh[LEFT_EYE_RIGHT][0]
            rlx = mesh[RIGHT_EYE_LEFT][0]; rrx = mesh[RIGHT_EYE_RIGHT][0]
            left_h = (l_iris[0] - llx) / (lrx - llx + 1e-6)
            right_h = (r_iris[0] - rlx) / (rrx - rlx + 1e-6)
            horiz_ratio = (left_h + right_h)/2.0

            # vertical normalized
            left_eye_h = abs(mesh[LEFT_TOP][1] - mesh[LEFT_BOTTOM][1]) + 1e-6
            right_eye_h = abs(mesh[RIGHT_TOP][1] - mesh[RIGHT_BOTTOM][1]) + 1e-6
            left_v = (l_iris[1] - mesh[LEFT_TOP][1]) / left_eye_h
            right_v = (r_iris[1] - mesh[RIGHT_TOP][1]) / right_eye_h
            vert_ratio = (left_v + right_v)/2.0

            # smoothing
            h_queue.append(horiz_ratio); v_queue.append(vert_ratio)
            if len(h_queue) > SMOOTH_LEN: h_queue.pop(0); v_queue.pop(0)
            sh, sv = float(np.mean(h_queue)), float(np.mean(v_queue))

            # baseline
            base_h, base_v = center_h, center_v
            dh, dv = sh - base_h, sv - base_v

            now = time.time()
            if dh < -HORIZ_THRESHOLD:
                gaze = "LEFT"
            elif dh > HORIZ_THRESHOLD:
                gaze = "RIGHT"
            elif dv < -VERT_THRESHOLD:
                gaze = "UP"
            elif dv > VERT_THRESHOLD:
                gaze = "DOWN"
            else:
                gaze = "CENTER"

            # navigation with cooldown
            if (now - last_action_time) > ACTION_DELAY:
                if gaze == "LEFT":
                    pyautogui.press('left'); last_action_time = now
                elif gaze == "RIGHT":
                    pyautogui.press('right'); last_action_time = now
                elif gaze == "UP":
                    pyautogui.scroll(300); last_action_time = now
                elif gaze == "DOWN":
                    pyautogui.scroll(-300); last_action_time = now

            # eye openness ratio
            l_top = mesh[LEFT_TOP]; l_bottom = mesh[LEFT_BOTTOM]
            r_top = mesh[RIGHT_TOP]; r_bottom = mesh[RIGHT_BOTTOM]
            l_corner_l = mesh[LEFT_EYE_LEFT]; l_corner_r = mesh[LEFT_EYE_RIGHT]
            r_corner_l = mesh[RIGHT_EYE_LEFT]; r_corner_r = mesh[RIGHT_EYE_RIGHT]

            left_open = abs(l_top[1] - l_bottom[1]) / (abs(l_corner_r[0] - l_corner_l[0]) + 1e-6)
            right_open = abs(r_top[1] - r_bottom[1]) / (abs(r_corner_r[0] - r_corner_l[0]) + 1e-6)
            open_ratio = (left_open + right_open)/2.0

            # adaptive baseline
            if len(open_history) < 60:
                open_history.append(open_ratio)
            else:
                open_history.pop(0); open_history.append(open_ratio)
            open_mean = float(np.mean(open_history)) if open_history else open_ratio
            CLOSED_THRESHOLD = open_mean * 0.6

            # blink detection (closed -> open transitions)
            if open_ratio < CLOSED_THRESHOLD and not blink_active:
                blink_active = True
                blink_start = time.time()
            elif open_ratio >= CLOSED_THRESHOLD and blink_active:
                blink_end = time.time()
                blink_active = False
                duration = blink_end - blink_start
                if duration >= MIN_BLINK_IGNORE:
                    if duration < DOT_DASH_SEP:
                        morse_buffer += '.'
                    else:
                        morse_buffer += '-'
                    last_blink_time = time.time()
                    char_commit_time = last_blink_time

            # commit morse char if pause
            if morse_buffer and (time.time() - last_blink_time) > WORD_PAUSE:
                decoded = decode_morse(morse_buffer)
                # append decoded char
                if decoded == '?':
                    typed_text += '?'
                else:
                    typed_text += decoded
                # reset buffer
                morse_buffer = ''
                # suggestions
                tokens = typed_text.strip().split()
                last_word = tokens[-1] if tokens else ''
                suggestions = predict_next(last_word) if last_word else []
                # update GUI-state safely via setting variables (main thread will read)
                # print debug
                print("[DECODED]", decoded, "| Suggestions:", suggestions)
        else:
            # no face landmarks detected: optional damping
            gaze = "NO_FACE"

        # overlay text and landmarks
        cv2.putText(frame, f"Gaze: {gaze}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"MorseBuf: {morse_buffer}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
        cv2.putText(frame, f"Typed: {typed_text}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # publish frame
        with frame_lock:
            latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()

# ---------------- GUI ----------------
class EyeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Eye-Control GUI - Rajat")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # top frame: video
        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, columnspan=4)

        # control buttons
        self.cal_btn = ttk.Button(root, text="Calibrate (C)", command=self.calibrate)
        self.cal_btn.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.clear_btn = ttk.Button(root, text="Clear Text", command=self.clear_text)
        self.clear_btn.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        self.typevar = tk.BooleanVar(value=False)
        self.typechk = ttk.Checkbutton(root, text="Type to App", variable=self.typevar)
        self.typechk.grid(row=1, column=2, sticky="ew", padx=4, pady=4)

        self.quit_btn = ttk.Button(root, text="Quit", command=self.on_close)
        self.quit_btn.grid(row=1, column=3, sticky="ew", padx=4, pady=4)

        # info labels
        self.gaze_var = tk.StringVar(value="Gaze: CENTER")
        self.gaze_label = ttk.Label(root, textvariable=self.gaze_var, font=("Segoe UI", 12))
        self.gaze_label.grid(row=2, column=0, sticky="w", padx=6)

        self.morse_var = tk.StringVar(value="MorseBuf: ")
        self.morse_label = ttk.Label(root, textvariable=self.morse_var)
        self.morse_label.grid(row=2, column=1, sticky="w", padx=6)

        self.typed_var = tk.StringVar(value="Typed: ")
        self.typed_label = ttk.Label(root, textvariable=self.typed_var)
        self.typed_label.grid(row=2, column=2, columnspan=2, sticky="w", padx=6)

        # suggestions area
        self.suggestion_frame = ttk.LabelFrame(root, text="Suggestions")
        self.suggestion_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        self.sugg_buttons = []
        for i in range(3):
            b = ttk.Button(self.suggestion_frame, text="", command=lambda i=i: self.apply_suggestion(i))
            b.grid(row=0, column=i, padx=6, pady=6, sticky="ew")
            self.sugg_buttons.append(b)

        # periodic update
        self.update_interval = 30  # ms
        self.update_gui()
        # start camera thread
        self.cam_thread = threading.Thread(target=camera_loop, daemon=True)
        self.cam_thread.start()

    def calibrate(self):
        # simple calibration: average current queue values
        global center_h, center_v, h_queue, v_queue, calibrated
        if len(h_queue) > 0:
            center_h = float(np.mean(h_queue))
            center_v = float(np.mean(v_queue))
            calibrated = True
            print(f"Calibrated center_h={center_h:.3f}, center_v={center_v:.3f}")
        else:
            center_h, center_v = 0.5, 0.5
            calibrated = True
            print("Calibrated to defaults.")

    def clear_text(self):
        global typed_text
        typed_text = ''
        self.typed_var.set("Typed: ")

    def apply_suggestion(self, idx):
        global typed_text
        # get last typed tokens and replace or append
        tokens = typed_text.strip().split()
        last_token = tokens[-1] if tokens else ''
        suggestions = predict_next(last_token) if last_token else []
        if idx < len(suggestions):
            sel = suggestions[idx]
            # append selection as next word
            typed_text += ' ' + sel + ' '
            if self.typevar.get():
                pyautogui.typewrite(sel + ' ')
            self.typed_var.set("Typed: " + typed_text)

    def update_gui(self):
        # update latest frame
        global latest_frame, morse_buffer, typed_text
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        if frame is not None:
            img = Image.fromarray(frame)
            img = img.resize((680, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # update text fields
        self.morse_var.set("MorseBuf: " + (morse_buffer if morse_buffer else ""))
        self.typed_var.set("Typed: " + typed_text)

        # update suggestions
        tokens = typed_text.strip().split()
        last_token = tokens[-1] if tokens else ''
        suggestions = predict_next(last_token) if last_token else []
        for i in range(3):
            txt = suggestions[i] if i < len(suggestions) else ''
            self.sugg_buttons[i].configure(text=txt if txt else "—")

        # auto-type recently committed char if typevar on
        # (we keep typing disabled for safety unless user toggles)
        # commit: handled in camera loop, but typing to app is done here
        if self.typevar.get():
            # naive approach: type full typed_text into app (not ideal for repeated calls)
            # safer: only type newly added content — we keep a simple pointer
            if not hasattr(self, 'last_typed_len'):
                self.last_typed_len = 0
            if len(typed_text) > self.last_typed_len:
                new_chunk = typed_text[self.last_typed_len:]
                pyautogui.typewrite(new_chunk)
                self.last_typed_len = len(typed_text)

        # schedule next update
        self.root.after(self.update_interval, self.update_gui)

    def on_close(self):
        global running
        running = False
        print("Shutting down...")
        self.root.after(200, self.root.quit)

# ---------------- Run App ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeGUI(root)
    root.mainloop()
    # cleanup
    running = False
    time.sleep(0.3)
    print("Exited.")
