import cv2
import datetime
import numpy as np
import time
import export_data
import requests

esp32_ip = "http://192.168.100.176/send-data"

def crop(frame, pts1, pts2):

    # Create a mask of the same size as the image, initialized with zeros (black)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

   # Fill the two polygons on the mask with white (255)
    cv2.fillPoly(mask, [pts1], 255)
    cv2.fillPoly(mask, [pts2], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_image

def calculate_pixel(frame):
    return np.std(frame)

def calculate_red_pixel_std(frame):
    # Extract the red channel (assuming BGR format)
    red_channel = frame[:, :, 2]
    
    # Filter out zero pixels (those not in the masked area)
    red_values = red_channel[red_channel > 0]
    
    # Calculate the standard deviation of the red channel pixels
    return np.std(red_values)

async def predict(model, img, ref_image, frame_count, ws, image_folder, conf=0.5, serial=''):
    crop_img = img.copy()
    results = model(img, conf=conf, verbose=False)
    if not results or len(results) == 0 and frame_count%1000 == 0:
        cv2.imwrite('ref_img.png', img)
        return img

    current_time = datetime.datetime.now().strftime("%I:%M%p")

    is_send = False
    for result in results:
        count = result.boxes.shape[0]
        for i in range(count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            if name == 'person':
                confidence = float(result.boxes.conf[i].item())
                bbox = result.boxes.xywh[i].cpu().numpy()
                x, y, w, h = bbox
                lx = int(x - w / 2)
                ux = int(x + w / 2)
                ly = int(y - h / 2)
                uy = int(y + h / 2)
                cv2.rectangle(img, (lx, ly), (ux, uy), (255, 0, 255), 1)
                cv2.putText(img, name + ':' + str(round(confidence, 2)), (int(bbox[0]), int(bbox[1] - 40)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                # Adjust this condition based on your cropping logic
                if lx >= 200 and ly >= 50 and uy <= 300:
                    pts1 = np.array([[256, 234], [258, 234], [305, 132], [303, 132]])
                    pts2 = np.array([[312, 113], [404, 92], [404, 94], [312, 115]])
                    crop_img = crop(crop_img, pts1, pts2)

                    pred_px = calculate_red_pixel_std(crop_img[ly:uy, lx:ux])
                    ref_px = calculate_red_pixel_std(ref_image[ly:uy, lx:ux])

                    # Check if the person is holding the handrail
                    if pred_px >= ref_px:
                        is_send = True
    if is_send:
        # Wifi
        response = requests.post(esp32_ip, data='on')
        # Serial USB
        # ser.write(b'on')
        export_data.write_to_excel(ws, image_folder, name, img, current_time, frame_count)
        time.sleep(3)

    return img