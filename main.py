import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import time

sensitivity = 30
greenLower = (60 - sensitivity, 100, 100)
greenUpper = (60 + sensitivity, 255, 255)


def main():
    vs = VideoStream(src=0).start() # www.pyimagesearch.com was referenced
    time.sleep(2.0)
    
    while(True):
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, greenLower, greenUpper)

        kernel = np.ones((5, 5), np.uint8)
        # Erode / dilate to make label appear "boxy"
        boxy = cv2.erode(mask, kernel, iterations=5)
        boxy = cv2.dilate(mask, kernel, iterations=5)

        im2, contours, hierarchy = cv2.findContours(
            boxy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            
            circ1 = 0
            circ2 = -1

            for i in range(len(contours) - 1):
                if cv2.contourArea(contours[i]) >= cv2.contourArea(contours[circ1]):
                    circ1 = i
                elif circ2 == -1 or cv2.contourArea(contours[i]) > cv2.contourArea(contours[circ2]):
                    circ2 = i

            # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
            
            for i in [circ1, circ2]:
                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

            print("Circle 1 Area: %d Circle 2 Area: %d" % (cv2.contourArea(contours[circ1]), cv2.contourArea(contours[circ2])))

            M1 = cv2.moments(contours[circ1])
            M2 = cv2.moments(contours[circ2])
            M1Com = (int(M1['m10']/M1['m00']), int(M1['m01']/M1['m00']))
            M2Com = (int(M2['m10']/M2['m00']), int(M2['m01']/M2['m00']))

            cv2.line(frame, M1Com, M2Com, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        cv2.imshow("boxy", boxy)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break


main()
