import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

cap = cv.VideoCapture('./data/surveillance.mpg')
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('Background.avi',fourcc, 30.0, (131,158))


fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True: # if return is true
        grame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(frame)
        cv.imshow('frame',fgmask)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv.destroyAllWindows()