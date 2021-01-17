import numpy as np
import cv2

from detector import *

background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
motion_detector = MyMotionDetector()
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    objects, mask = motion_detector(frame)

    print(len(objects))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('fgMask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()