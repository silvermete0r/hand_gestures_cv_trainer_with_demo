import os
import pickle
import mediapipe as mp
import cv2
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

# Paths
DATA_DIR = "data"
IMG_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)

data = []
labels = []

# Traverse image directories
label_dirs = [label for label in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, label))]
for label in tqdm(label_dirs, desc="Processing directories", unit="dir"):
    label_path = os.path.join(IMG_DIR, label)

    img_files = os.listdir(label_path)
    for img_name in tqdm(img_files, desc=f"Processing images in {label}", unit="img", leave=False):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_, y_ = [], []

                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(label)

# Save the data and labels as a single pickle file
file_path = os.path.join(DATA_DIR, "hand_landmarks_data.pkl")
with open(file_path, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"Saved {len(data)} samples to {file_path}")
