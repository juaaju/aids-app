import cv2 as cv
import numpy as np


def empty(a):
    pass

def initialize_trackbar():
    cv.namedWindow('HSV')
    cv.resizeWindow('HSV', 640, 240)
    cv.createTrackbar('Hue Min', 'HSV', 0, 179, empty)
    cv.createTrackbar('Hue Max', 'HSV', 179, 179, empty)
    cv.createTrackbar('Sat Min', 'HSV', 0, 255, empty)
    cv.createTrackbar('Sat Max', 'HSV', 255, 255, empty)
    cv.createTrackbar('Val Min', 'HSV', 0, 255, empty)
    cv.createTrackbar('Val Max', 'HSV', 255, 255, empty)

def get_trackbar_positions():
    h_min = cv.getTrackbarPos('Hue Min', 'HSV')
    h_max = cv.getTrackbarPos('Hue Max', 'HSV')
    s_min = cv.getTrackbarPos('Sat Min', 'HSV')
    s_max = cv.getTrackbarPos('Sat Max', 'HSV')
    v_min = cv.getTrackbarPos('Val Min', 'HSV')
    v_max = cv.getTrackbarPos('Val Max', 'HSV')

    return np.array([h_min, s_min, v_min, h_max, s_max, v_max])

def detect_color_img(img):
    initialize_trackbar()

    #by image
    while True:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        pos = get_trackbar_positions()

        lower = np.array([pos[0], pos[1], pos[2]])
        upper = np.array([pos[3], pos[4], pos[5]])
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(img, img, mask=mask)

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        h_stack = np.hstack([img, mask, result])

        cv.imshow("Result", h_stack)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

def detect_color_vid(cap):
    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        if cap.get(cv.CAP_PROP_FRAME_COUNT) == frame_count:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

        ret, frame = cap.read()
        frame = cv.resize(frame, (480, 240))
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        pos = get_trackbar_positions()

        lower = np.array([pos[0], pos[1], pos[2]])
        upper = np.array([pos[3], pos[4], pos[5]])
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(frame, frame, mask=mask)

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        h_stack = np.hstack([frame, mask, result])
        cv.imshow('result', h_stack)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Load the image from a file
image_path = 'test.png'
img = cv.imread(image_path)

# Ensure the image is successfully loaded
if img is None:
    print("Error: Image not found.")
else:
    detect_color_img(img)