import cv2
import mediapipe as mp


class hand_detection:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5, draw=False, give_data=True):
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
            for handLms in results.multi_hand_landmarks:
                if self.giveData:
                    self.give_hand_data(frame, handLms)

                if self.draw:
                    self.draw_hand(frame, handLms)

    # will draw hand
    def draw_hand(self, frame, hand_lms):
        # drawing hand dots and connections
        self.mpDraw.draw_landmarks(frame, hand_lms, self.mpHands.HAND_CONNECTIONS)

    # will give data
    def give_hand_data(self, frame, hand_lms):
        self.handList.clear()
        for (ID, lm) in enumerate(hand_lms.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            self.handList.append({
                "id": ID,
                "centerX": cx,
                "centerY": cy
            })


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
