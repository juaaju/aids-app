import numpy as np
import cv2
import matplotlib.pyplot as plt

def crop(frame, pts1, pts2):
    frame = cv2.resize(frame,(416, 416))
    # Create a mask of the same size as the image, initialized with zeros (black)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

   # Fill the two polygons on the mask with white (255)
    cv2.fillPoly(mask, [pts1], 255)
    cv2.fillPoly(mask, [pts2], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # # Find the bounding box that includes both polygons
    # combined_pts = np.vstack((pts1, pts2))  # Combine both sets of points
    # x, y, w, h = cv2.boundingRect(combined_pts)

    # # Crop the image using the bounding box
    # cropped_image = masked_image[y:y+h, x:x+w]
    # cropped_image = cv2.resize(cropped_image,(416, 416))

    return masked_image

image = cv2.imread('sample_image.png')
pts1 = np.array([[256, 234], [258, 234], [305, 132], [303, 132]])
pts2 = np.array([[312, 113], [404, 92], [404, 94], [312, 115]])

crop_image = crop(image, pts1, pts2)
cv2.imwrite('clean_handrail.png', crop_image)
cv2.imshow('image', crop_image)
cv2.waitKey(0)

# plt.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
# plt.show()


# cap = cv2.VideoCapture('test.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         # Display the resulting frame
#         cv2.imshow('frame', crop(frame, pts1, pts2))

#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# plt.imshow(cv2.cvtColor(crop(image, pts1, pts2), cv2.COLOR_BGR2RGB))
# # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()