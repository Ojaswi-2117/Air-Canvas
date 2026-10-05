import cv2
import numpy as np
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Canvas
canvas = None

# Previous point
prev_x, prev_y = 0, 0

# Default color (blue)
color = (255, 0, 0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw top menu
    cv2.rectangle(frame, (0, 0), (640, 60), (50, 50, 50), -1)

    cv2.putText(frame, "BLUE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "GREEN", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "RED", (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "YELLOW", (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "ERASER", (460, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (560, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Instructions
    cv2.putText(frame, "1 Finger = Draw | 2 Fingers = Select", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            # Index finger tip
            x1 = int(lm[8].x * frame.shape[1])
            y1 = int(lm[8].y * frame.shape[0])

            # Middle finger tip
            x2 = int(lm[12].x * frame.shape[1])
            y2 = int(lm[12].y * frame.shape[0])

            # Finger states
            fingers_up = []

            # Index
            fingers_up.append(lm[8].y < lm[6].y)
            # Middle
            fingers_up.append(lm[12].y < lm[10].y)

            # ✌️ Selection mode (2 fingers)
            if fingers_up[0] and fingers_up[1]:
                prev_x, prev_y = 0, 0

                if y1 < 60:
                    if 10 < x1 < 100:
                        color = (255, 0, 0)
                    elif 110 < x1 < 200:
                        color = (0, 255, 0)
                    elif 230 < x1 < 320:
                        color = (0, 0, 255)
                    elif 330 < x1 < 440:
                        color = (0, 255, 255)
                    elif 460 < x1 < 550:
                        color = (0, 0, 0)  # eraser
                    elif 560 < x1 < 640:
                        canvas = np.zeros_like(frame)

                cv2.rectangle(frame, (x1, y1-20), (x2, y2+20), (255, 255, 255), 2)

            # ☝️ Drawing mode (1 finger)
            elif fingers_up[0] and not fingers_up[1]:
                cv2.circle(frame, (x1, y1), 8, color, -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                thickness = 20 if color == (0,0,0) else 5
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), color, thickness)

                prev_x, prev_y = x1, y1

    else:
        prev_x, prev_y = 0, 0

    # Merge canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Canvas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()