import cv2
import mediapipe as mp


class hand_detection:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5, draw=False,
                 give_data=True):
        self.mode = mode
        self.maxHands = max_hands
        self.detectionConfidence = detection_confidence
        self.trackingConfidence = tracking_confidence
        self.draw = draw
        self.giveData = give_data

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackingConfidence)
        # getting drawing utilities for our detected hands
        self.mpDraw = mp.solutions.drawing_utils
        self.handList = []

    def detect_hand(self, frame):
        # converting to rgb because it only detect in rgb image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # processing frame to find hand in it
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            # print("###########################")
            for handLms in results.multi_hand_landmarks:  # looping to hand
                if self.giveData:
                    self.give_hand_data(frame, handLms)

                if self.draw:
                    self.draw_hand(frame, handLms)

    # will draw hand
    def draw_hand(self, frame, hand_lms):
        # drawing hand dots and connections
        return self.mpDraw.draw_landmarks(frame, hand_lms, self.mpHands.HAND_CONNECTIONS)

    # will give data
    def give_hand_data(self, frame, hand_lms):
        self.handList.clear()
        h, w, c = frame.shape

        for (ID, lm) in enumerate(hand_lms.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)

            self.handList.append({
                "id": ID,
                "centerX": cx,
                "centerY": cy,
            })

    # Will give which finger is up
    def give_finger_status(self):
        pointed_fingers = []
        n_pointed_fingers = []

        # checking if thumb finger is up
        if self.handList[4]['centerY'] > self.handList[3]['centerY']:
            n_pointed_fingers.append(4)
        else:
            pointed_fingers.append(4)

        # checking if index finger is up
        if self.handList[8]['centerY'] > self.handList[7]['centerY']:
            n_pointed_fingers.append(8)
        else:
            pointed_fingers.append(8)

        # checking if middle finger is up
        if self.handList[12]['centerY'] > self.handList[11]['centerY']:
            n_pointed_fingers.append(12)
        else:
            pointed_fingers.append(12)

        # checking if ring finger is up
        if self.handList[16]['centerY'] > self.handList[15]['centerY']:
            n_pointed_fingers.append(16)
        else:
            pointed_fingers.append(16)

        # checking if pinky finger is up
        if self.handList[20]['centerY'] > self.handList[19]['centerY']:
            n_pointed_fingers.append(20)
        else:
            pointed_fingers.append(20)

        return pointed_fingers, n_pointed_fingers

    def give_distance(self, axis, finger1_index, finger2_index):
        if axis == 'x' or axis == 'X':
            return self.handList[finger1_index]['centerX'] - self.handList[finger2_index]['centerX']
        elif axis == 'y' or axis == 'Y':
            return self.handList[finger1_index]['centerY'] - self.handList[finger2_index]['centerY']
        else:
            print("Error: invalid axis")


def main():
    camera = cv2.VideoCapture(0)

    det = hand_detection()

    while True:
        ret, frame = camera.read()

        det.detect_hand(frame)
        print(det.handList)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
