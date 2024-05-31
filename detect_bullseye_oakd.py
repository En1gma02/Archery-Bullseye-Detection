import cv2
import pickle
import depthai as dai

# Load the trained model
with open('bullseye_detector.pkl', 'rb') as f:
    model = pickle.load(f)

lower_yellow = np.array(model['lower_yellow'])
upper_yellow = np.array(model['upper_yellow'])


def extract_yellow_region(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask


# Initialize Oak-D Lite camera
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
xout_video = pipeline.createXLinkOut()

cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
xout_video.setStreamName("video")

cam_rgb.video.link(xout_video.input)

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        video_frame = video_queue.get()
        frame = video_frame.getCvFrame()

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

                # Add your payload drop logic here using (cx, cy)
                print(f"Bullseye center detected at: ({cx}, {cy})")

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
