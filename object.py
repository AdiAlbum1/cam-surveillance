import numpy as np

class Object:
    def __init__(self):
        self.centroid = np.array([0, 0])            # coordinates: x,y
        self.roi = [0, 0, 0, 0]                     # crop region: x, y, w, h
        self.area = 0                               # area in pixels

    def is_valid_object(self):
        return self.area > 400