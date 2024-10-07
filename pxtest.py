import numpy as np
import cv2

# # Load the image using OpenCV
# image = cv2.imread('clean_handrail.jpg')

# # # Optional: Convert the image to grayscale (if you want standard deviation of intensity values)
# # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # Flatten the image pixels into a single array
# # pixel_values = gray_image.flatten()

# # Calculate the standard deviation using NumPy
# std_dev = np.std(image)

# print("Standard Deviation of pixel values:", std_dev)

# def crop(frame, pts1, pts2, bbox):

#     x,y,w,h = bbox

#     # Create a mask of the same size as the image, initialized with zeros (black)
#     mask = np.zeros(frame.shape[:2], dtype=np.uint8)

#    # Fill the two polygons on the mask with white (255)
#     cv2.fillPoly(mask, [pts1], 255)
#     cv2.fillPoly(mask, [pts2], 255)

#     # Apply the mask to the image
#     masked_image = cv2.bitwise_and(frame, frame, mask=mask)

#     cv2.fillPoly(masked_image, [np.array([[int(x-w/2), int(y-h/2)], [int(x+w/2), int(y-h/2)], [int(x+w/2), int(y+h/2)], [int(x-w/2), int(y+h/2)]])], 255)

#     masked_image = cv2.bitwise_and(frame, frame, mask=mask)

#     return masked_image

# image = cv2.imread('frame_image_75.png')

# bbox = [300, 60, 30, 80]
# pts1 = np.array([[255, 234], [260, 234], [307, 130], [302, 131]])
# pts2 = np.array([[311, 113], [404, 91], [404, 95], [311, 117]])
# crop_img = crop(image, pts1, pts2, bbox)

# cv2.imshow('frame', crop_img)
# cv2.waitKey(0)

image = cv2.imread('sample_image.png')
crop_img = image[0:300, 0:300]

cv2.imshow('frame', crop_img)
cv2.waitKey(0)