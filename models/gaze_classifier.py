# models/gaze_classifier.py

class GazeClassifier:
    def __init__(self, horiz_threshold=0.18, vert_threshold=0.18):
        self.horiz_th = horiz_threshold
        self.vert_th = vert_threshold

    def classify(self, dh, dv):
        if dh < -self.horiz_th:
            return "LEFT"
        if dh > self.horiz_th:
            return "RIGHT"
        if dv < -self.vert_th:
            return "UP"
        if dv > self.vert_th:
            return "DOWN"
        return "CENTER"
