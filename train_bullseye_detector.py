import cv2
import pickle
import numpy as np
import os


def extract_yellow_region(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    return mask


def train_detector(image_folder):
    # Read all images and process them
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)
            mask = extract_yellow_region(image)

            # Display the mask for verification
            cv2.imshow('Mask', mask)
            cv2.waitKey(100)  # Adjust the delay as needed

    cv2.destroyAllWindows()

    # Save the color threshold values
    model = {'lower_yellow': [20, 100, 100], 'upper_yellow': [30, 255, 255]}
    with open('bullseye_detector.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    image_folder = 'E:/Phoenix/SAE target/Dataset'
    train_detector(image_folder)
