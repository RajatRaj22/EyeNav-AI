# models/blink_detector.py

import time

class BlinkDetector:
    def __init__(self, dot_dash_sep=0.45, min_ignore=0.06):
        self.dot_dash_sep = dot_dash_sep
        self.min_ignore = min_ignore
        self.blink_active = False
        self.blink_start = 0.0
        self.last_blink_time = 0.0

    def update(self, open_ratio, closed_threshold):
        now = time.time()

        # Eye goes closed
        if open_ratio < closed_threshold and not self.blink_active:
            self.blink_active = True
            self.blink_start = now

        # Eye reopens → blink done
        elif open_ratio >= closed_threshold and self.blink_active:
            self.blink_active = False
            duration = now - self.blink_start
            if duration < self.min_ignore:
                return None
            self.last_blink_time = now
            return '.' if duration < self.dot_dash_sep else '-'

        return None
