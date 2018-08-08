import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import time


sensitivity = 20
greenLower = (60 - sensitivity, 100, 100)
greenUpper = (60 + sensitivity, 255, 255)


def main():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    while(True):
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, greenLower, greenUpper)

        kernel = np.ones((5, 5), np.uint8)
        boxy = cv2.erode(mask, kernel, iterations = 5) #Erode / dilate to make label appear "boxy"
        boxy = cv2.dilate(mask, kernel, iterations = 5)


        im2, contours, hierarchy = cv2.findContours(boxy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        # (x,y),radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x),int(y))
        # radius = int(radius)
        # cv2.circle(frame,center,radius,(0,255,0),2)

        cv2.imshow("Frame", frame)
        cv2.imshow("boxy", boxy)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break





main()