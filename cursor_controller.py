import cv2
import numpy as np
from hand_detector import hand_detection
import autopy
import os

# cam resolution
wCam, hCam = 640, 480
# screen resolution
wScreen, hScreen = autopy.screen.size()
# frame reduction
frameR = 100
# smoothing
smoothing = 2
# current x and y value
cXvalue, cYvalue = 0, 0
# previous x and y value
pXvalue, pYvalue = 0, 0

# instantiate class
hand_detector_inst = hand_detection(max_hands=1)

# initializing camera
camera = cv2.VideoCapture(0)
camera.set(3, wCam)
camera.set(4, hCam)

# status[ideal,moving,clicking]
status = 0

while True:
    ret, frame = camera.read()

    # WARNING:: i have commented this because it will cause lag #

    # if status == 0:
    #     cv2.putText(frame, "Ideal", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    # elif status == 1:
    #     cv2.putText(frame, "Moving", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    # elif status == 2:
    #     cv2.putText(frame, "Clicking", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # detecting hand
    hand_detector_inst.detect_hand(frame)
    # extracting hand data
    hand_data = hand_detector_inst.handList

    # checking if hand data isn't null
    if len(hand_data) > 0:
        # checking which fingers are up
        pointed_finger, n_pointed_finger = hand_detector_inst.give_finger_status()
        # extracting index and middle finger tip data
        index_finger_tip, middle_finger_tip = hand_detector_inst.handList[8], hand_detector_inst.handList[12]

        # creating a limit frame
        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 50, 50), 2)

        # checking if index finger is up or index finger and thumb are up
        if pointed_finger == [8] or pointed_finger == [4, 8]:
            # converting x and y coordinates from webcam resolution to screen resolution
            x = np.interp(index_finger_tip['centerX'], (frameR, wCam - frameR), (0, wScreen))
            y = np.interp(index_finger_tip['centerY'], (frameR, hCam - frameR), (0, hScreen))

            # this is another way to convert coordinate from one res to another ( new_x = old_x / old_res_width * new_res_width)
            # x = index_finger_tip['centerX'] / wCam * wScreen
            # y = index_finger_tip['centerY'] / hCam * hScreen

            # smoothing
            cXvalue = pXvalue + (x - pXvalue) / smoothing
            cYvalue = pYvalue + (y - pYvalue) / smoothing

            autopy.mouse.move(wScreen - cXvalue, cYvalue)

            pXvalue, pYvalue = cXvalue, cYvalue
            status = 1
        elif pointed_finger == [] or pointed_finger == [4]:
            # click
            autopy.mouse.click()
            status = 2

    # cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
