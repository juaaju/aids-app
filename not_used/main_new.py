import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt

def detect_people(model, image):
    # Perform object detection using YOLOv8
    results = model(image)

    # Extract person bounding boxes
    person_boxes = []
    for result in results:  # Loop over each detection
        bboxes = result.boxes
        for bbox in bboxes:
            b = bbox.xyxy[0]
            c = bbox.cls
            if c == 0:  # Check if the detected class is a person
                x1, y1, x2, y2 = map(int, b[:4])
                person_boxes.append([x1, y1, x2, y2])
    return bboxes, person_boxes

def draw_bboxes(bboxes, image):
    annotator = Annotator(image)
    for bbox in bboxes:
        b = bbox.xyxy[0]
        c = bbox.cls
        name = model.names[int(c)]
        annotator.box_label(b, name, (255, 0, 255))
    return image 

def thresholding(img, h_max, h_min, v_max, v_min, s_max, s_min):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

# Analyze the handrail to see if it is being "cut" by any person’s hand
def is_handrail_held(person_boxes, handrail_mask):
    for box in person_boxes:
        x1, y1, x2, y2 = box
        # Extract the part of the mask corresponding to the person's bounding box
        person_region = handrail_mask[y1:y2, x1:x2]

        # Sum the pixels along the handrail in the bounding box region
        handrail_sum = np.sum(person_region == 255)

        # If there is a significant reduction in the white pixels, it indicates the handrail is being held
        if handrail_sum > 0:
            print(f"Person at {box} is holding the handrail.")
        else:
            print(f"Person at {box} is NOT holding the handrail.")

# Load the image
image_path = 'test.png'
image = cv2.imread(image_path)

# Load the YOLOv8 model (assuming you have the pretrained weights for person detection)
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' or another version

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform segmentation to detect the handrail
# You can adjust the threshold based on the color of the handrail in the image
# _, handrail_mask = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

thresh_image = thresholding(image, )

cv2.imshow('gray', thresh_image)
cv2.waitKey(0)

# Find contours in the segmented handrail
contours, _ = cv2.findContours(handrail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)

# Draw the contours of the handrail for visualization
cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

cv2.imshow('cc', image)
cv2.waitKey(0)

# # Detect people on the stairs
# person_boxes = detect_people(model, image)

# # Check for each person if they are holding the handrail
# is_handrail_held(person_boxes, handrail_mask)

# # Show the final image with the detections
# # cv2.imshow('Handrail Detection', image)
# # cv2.waitKey(0)
# plt.imshow(image)
# plt.show()
# cv2.destroyAllWindows()
