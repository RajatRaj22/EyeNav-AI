# utils/filters.py

class SmoothQueue:
    def __init__(self, size):
        self.size = size
        self.queue = []

    def add(self, val):
        self.queue.append(val)
        if len(self.queue) > self.size:
            self.queue.pop(0)

    def mean(self):
        if not self.queue:
            return 0
        return sum(self.queue) / len(self.queue)
