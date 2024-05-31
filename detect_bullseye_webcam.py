import cv2
import pickle
import numpy as np

# Load the trained model
with open('bullseye_detector.pkl', 'rb') as f:
    model = pickle.load(f)

lower_yellow = np.array(model['lower_yellow'])
upper_yellow = np.array(model['upper_yellow'])


def extract_yellow_region(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    mask = extract_yellow_region(frame)
    cv2.imshow('Mask', mask)

    # Find contours and detect the center of the yellow region
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw the center point
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            print(f"Bullseye center detected at: ({cx}, {cy})")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
