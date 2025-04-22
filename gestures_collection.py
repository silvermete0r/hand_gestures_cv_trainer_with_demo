import os
import cv2

IMG_DIR = "output"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

labels = [
    # "A",
    # "B",
    # "C",
    # "D",
    # "E",
    # "F",
    # "G",
    # "H",
    # "I",
    # "J",
    # "K",
    # "L",
    # "M",
    # "N",
    # "O",
    # "P",
    # "Q",
    # "R",
    # "S",
    # "T",
    # "U",
    # "V",
    # "W",
    # "X",
    # "Y",
    # "Z",
    "Good",
    "Bad"
]

number_of_samples = 200

cap = cv2.VideoCapture(0) # 0 for default camera

for i in range(len(labels)):
    label = labels[i]
    if not os.path.exists(os.path.join(IMG_DIR, label)):
        os.makedirs(os.path.join(IMG_DIR, label))

    print(f"#{i+1}: Collecting samples for {label}...")
    print("Press 'S' to start collecting samples.")
    print("Press 'Q' to quit the collection process.")
    
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"Before starting: {label} | Press 'S' to start!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow('Gesture Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Starting collection...")
            break
        elif key == ord('q'):
            print("Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cnt = 0
    while cnt < number_of_samples:
        ret, frame = cap.read()
        cv2.imshow('Gesture Collection', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(IMG_DIR, label, f"{label}_{cnt}.jpg"), frame)
        cnt += 1
        print(f"Collected {cnt}/{number_of_samples} samples for {label}...", end="\r")

cap.release()
cv2.destroyAllWindows()
print("\nCollection complete!")
