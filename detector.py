import cv2
from object import *

class MyMotionDetector:
    class Params:
        def __init__(self):
            self.close_size = 7
            self.open_size = 3
            self.median = 7
            self.history = 500
            self.detect_shadows = True
            self.connectivity = 4

    def __init__(self, params=Params()):
        self._params = params
        self._bgs = cv2.createBackgroundSubtractorMOG2(history=self._params.history, detectShadows=self._params.detect_shadows)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self._params.open_size, self._params.open_size))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._params.close_size, self._params.close_size))

    def __call__(self, frame):
        # Apply motion detection algorithm
        mask = self._bgs.apply(frame)
        # Threshold to keep only foreground: 0 - background, 127 - shadow, 255 - foreground
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # Filter noise using Median filter and morphological operations Open & Close.
        mask = ((cv2.boxFilter(mask,cv2.CV_8U,(self._params.median,self._params.median)) > 255*0.3)*255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)  # remove small objects
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)  # fill the holes

        # Connected Components Algorithm
        num_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=self._params.connectivity)

        # Define our list of objects and insert every valid object
        objects = []
        for n in range(1, num_comp):
            new_object = Object()
            new_object.roi = stats[n, :4]
            new_object.area = stats[n,4]
            new_object.centroid = centroids[n:]

            if (new_object.is_valid_object()):
                objects.append(new_object)

        return objects, mask