# utils/helpers.py

import numpy as np

def mean_point(points):
    pts = np.array(points)
    return np.mean(pts, axis=0).astype(int)
