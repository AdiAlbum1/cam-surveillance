import cv2

class MyObjectClassifier:
    class Params:
        def __init__(self):
            self.height = 256
            self.width = 128
    def __init__(self, params=Params()):
        pass

    def __call__(self, frame, objects):
        for object in objects:
            x, y, w, h = object.roi
            object_crop = frame[y:y+h, x:x+w]
            cv2.imshow("crop", object_crop)
            if cv2.waitKey(0) & 0xFF == ord('a'):
                break
            cv2.destroyWindow("crop")