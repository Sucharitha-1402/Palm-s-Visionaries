
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define a mock AI model to generate descriptive narratives for recognized gestures
def generate_narrative(gesture):
    narratives = {
        "thumbs_up": "You gave a thumbs up! Great job!",
        "fist": "You made a fist. Ready for action!",
        "open_hand": "Your hand is open. Hello there!",
        "peace": "Peace sign detected. Spread love and peace!"
    }
    return narratives.get(gesture, "Gesture not recognized.")

# Define the gesture recognition function
def recognize_gesture(landmarks):
    thumb_is_open = landmarks[4].y < landmarks[3].y < landmarks[2].y
    index_is_open = landmarks[8].y < landmarks[6].y
    middle_is_open = landmarks[12].y < landmarks[10].y
    ring_is_open = landmarks[16].y < landmarks[14].y
    pinky_is_open = landmarks[20].y < landmarks[18].y

    if thumb_is_open and not (index_is_open or middle_is_open or ring_is_open or pinky_is_open):
        return "thumbs_up"
    elif not (thumb_is_open or index_is_open or middle_is_open or ring_is_open or pinky_is_open):
        return "fist"
    elif all([thumb_is_open, index_is_open, middle_is_open, ring_is_open, pinky_is_open]):
        return "open_hand"
    elif index_is_open and middle_is_open and not (ring_is_open or pinky_is_open):
        return "peace"
    else:
        return "unknown"

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks.landmark)
            narrative = generate_narrative(gesture)
            cv2.putText(frame, narrative, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Gesture-Based Human-Computer Interaction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()