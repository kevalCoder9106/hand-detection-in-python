import cv2
import hand_detection

detector = hand_detection.hand_detection(max_hands=4, draw=True)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    detector.detect_hand(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
