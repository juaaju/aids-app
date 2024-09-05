import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from threading import Thread
from gtts import gTTS
from playsound import playsound
import datetime
import openpyxl
from openpyxl import Workbook
import datatodb
import asyncio
from bleak import BleakClient

class CamStream:
    def __init__(self, stream_id=0):
        self.vcap = cv2.VideoCapture(stream_id)
        if not self.vcap.isOpened():
            raise RuntimeError("Error accessing stream.")
        
        print(f"FPS of hardware/input stream: {int(self.vcap.get(cv2.CAP_PROP_FPS))}")
        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            raise RuntimeError("No more frames to read")
        
        self.stopped = True
        self.t = Thread(target=self.update, daemon=True)

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.vcap.read()
            if not self.grabbed:
                print("No more frames to read")
                self.stopped = True
        self.vcap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def crop_image(img, pts):
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, 255, -1, cv2.LINE_AA)
    return cv2.bitwise_and(cropped, cropped, mask=mask)

def predict(model, img, conf=0.5):
    results = model(img, conf=conf)
    if not results:
        return img
    current_time = datetime.datetime.now().strftime("%I:%M%p")
    for result in results:
        count=result.boxes.shape[0]
        for i in range(count):
            cls=int(result.boxes.cls[i].item())
            name=result.names[cls]
            confidence=float(result.boxes.conf[i].item())
            bbox = result.boxes.xywh[i].cpu().numpy()
            x,y,w,h = bbox
            cv2.rectangle(img, (int(x-w/2), int(y-h/2)),(int(x+w/2), int(y+h/2)),(255,0,255),1)
            cv2.putText(img, f"{name}:{round(confidence, 2)}", (int(x), int(y-40)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            write_to_excel(name, img, current_time)
            text = is_target_object(name)
            if text != '':
                sound_notification(text)
    return (), img

def is_target_object(obj):
    if obj in ['nohandrailmidleft', 'nohandrailleftfar', 'nohandrailmidright', 'nohandrailrightfar', 'nohandrailupleft', 'nohandraillowright']:
        return 'mohon pegang handrail'
    return ""

def sound_notification(text):
    tts = gTTS(text, lang='id', slow=False)
    tts.save('speech.mp3')
    playsound('speech.mp3')

async def connect_bluetooth(address, model):
    async with BleakClient(address) as client:
        model_number = await client.read_gatt_char(model)
        print("Model Number: {0}".format("".join(map(chr, model_number))))

async def send_bluetooth_signal(command, address):
    async with BleakClient(address) as client:
        await client.write_gatt_char('0000ffe1-0000-1000-8000-00805f9b34fb', command.encode())

async def light_notification():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_bluetooth_signal('1'))
    await asyncio.sleep(3)
    await send_bluetooth_signal('0')

def write_to_excel(data, current_time):
    img = openpyxl.drawing.image.Image('test.jpg')
    ws.append([data, current_time, img])

video_path = "rtsp://admin:pertamina321@10.205.64.111:554/Streaming/Channels/301"
model = YOLO('yolov8n.pt')

# Device address and model number. Use bluetooth_scanner.py to find pairable devices.
# address = "24:71:89:cc:09:05"
# uuid= "2A24"
# asyncio.run(connect_bluetooth(address, uuid))

pts = np.array([[286, 356], [236, 250], [293, 106], [416, 76], [416, 105], [326, 135], [274, 255], [311, 346]])

cam_stream = CamStream(0)
cam_stream.start()

wb = Workbook()
ws = wb.active

frame_processed = 0
start = time.time()

while True:
    if cam_stream.stopped:
        break
    frame = cam_stream.read()
    frame = cv2.resize(frame, (416, 416))
    frame = predict(model, frame)
    frame_processed += 1
    cv2.imshow('cctv', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_stream.stop()
cv2.destroyAllWindows()

wb.save('handrail.xlsx')
# datatodb.write_to_db()
