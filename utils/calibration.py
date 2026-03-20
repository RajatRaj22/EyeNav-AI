# utils/calibration.py

class Calibration:
    def __init__(self):
        self.center_h = 0.5
        self.center_v = 0.5

    def calibrate(self, sh, sv):
        self.center_h = sh
        self.center_v = sv

    def delta(self, sh, sv):
        return sh - self.center_h, sv - self.center_v
