import numpy as np
import cv2

from detector.detector import *
from classifier.classifier import *

# Load Video
cap = cv2.VideoCapture("example/input.avi")
# cap = cv2.VideoCapture(0)

# Initialize Motion Detection Module
motion_detector = MyMotionDetector()
classifier = MyObjectClassifier()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Apply Motion Detection
    objects, mask = motion_detector(frame)
    classifier(frame, objects)

    # Draw bounding box
    for obj in objects:
        x,y,w,h = obj.roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('fgMask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()