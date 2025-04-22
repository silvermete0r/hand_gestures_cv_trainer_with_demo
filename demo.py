import cv2
import mediapipe as mp
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load model and label encoder
model = joblib.load("xgb_hand_gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Gesture Demo", cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_label = "No hand detected"
    box_coords = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_ = [], []
            data_aux = []

            for lm in hand_landmarks.landmark:
                x_.append(int(lm.x * w))
                y_.append(int(lm.y * h))

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(lm.x for lm in hand_landmarks.landmark))
                data_aux.append(lm.y - min(lm.y for lm in hand_landmarks.landmark))

            # Predict label
            prediction = model.predict([data_aux])[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Bounding box
            x_min, x_max = min(x_), max(x_)
            y_min, y_max = min(y_), max(y_)
            box_coords = (x_min, y_min, x_max, y_max)

            break  # Only one hand

    # Draw label near bounding box
    if box_coords:
        x_min, y_min, x_max, y_max = box_coords
        label_pos = (x_min, y_min - 10 if y_min - 10 > 20 else y_min + 30)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
        cv2.putText(frame, predicted_label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Hand Gesture Demo", frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # ESC or Q to quit
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
