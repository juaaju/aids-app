import base64
import cv2
import numpy as np
from playsound import playsound
import datetime
from threading import Thread, Lock

# Global sound lock dan timer
sound_lock = Lock()
last_sound_time = datetime.datetime.now()
MIN_SOUND_INTERVAL = 2  # minimal interval dalam detik antara suara

def crop(frame, pts1, pts2):
    # Create a mask of the same size as the image, initialized with zeros (black)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
   # Fill the two polygons on the mask with white (255)
    cv2.fillPoly(mask, [pts1], 255)
    cv2.fillPoly(mask, [pts2], 255)
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_image

def calculate_iou(coords1, coords2):
    print(coords1)
    print(coords2)
    x1, y1 = max(coords1[0], coords2[0]), max(coords1[2], coords2[2])
    x2, y2 = min(coords1[1], coords2[1]), min(coords1[3], coords2[3])

    inter_width, inter_height = max(0, x2 - x1), max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = (coords1[1] - coords1[0]) * (coords1[3] - coords1[2])
    area2 = (coords2[1] - coords2[0]) * (coords2[3] - coords2[2])

    union_area = area1 + area2 - inter_area
    print(union_area)
    iou = inter_area/union_area if union_area!=0 else 0
    return iou

def calculate_red_pixel_std(frame):
    # Extract the red channel (assuming BGR format)
    red_channel = frame[:, :, 2]
    
    # Filter out zero pixels (those not in the masked area)
    red_values = red_channel[red_channel > 0]
    
    # Calculate the standard deviation of the red channel pixels
    return np.std(red_values)

def save_frame(frame):
    _, im_arr = cv2.imencode('.jpg', frame)
    im_b64 = base64.b64encode(im_arr)
    return im_b64.decode('utf-8')

# Buat fungsi untuk memainkan sound secara async
def play_sound_async(sound_file):
    global last_sound_time
    
    # Cek apakah sudah cukup waktu sejak suara terakhir
    current_time = datetime.datetime.now()
    with sound_lock:
        if (current_time - last_sound_time).total_seconds() < MIN_SOUND_INTERVAL:
            return  # Skip jika belum cukup waktu
        
        # Update waktu terakhir suara diputar
        last_sound_time = current_time
        
        # Putar suara dalam thread terpisah
        def play():
            playsound(sound_file)
        
        sound_thread = Thread(target=play)
        sound_thread.daemon = True
        sound_thread.start()
        print("sound diputar")