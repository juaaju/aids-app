import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from threading import Thread
import time
from gtts import gTTS
from playsound import playsound
import datetime
import openpyxl
from openpyxl import Workbook
import datatodb
import asyncio
from bleak import BleakClient

# defining a helper class for implementing multi-threaded processing 
class CamStream :
    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream 
        self.stopped = True 

        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
    # method for starting the thread for grabbing next available frame in input stream 
    def start(self):
        self.stopped = False
        self.t.start() 

    # method for reading next frame 
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True

def crop_image(img, pts):
	rect=cv2.boundingRect(pts)
	x,y,w,h=rect
	cropped = img[y:y+h,x:x+w].copy()
	pts=pts-pts.min(axis=0)
	mask=np.zeros(cropped.shape[:2],np.uint8)
	cv2.drawContours(mask, [pts], -1, (255,255,255), -1, cv2.LINE_AA)
	dst = cv2.bitwise_and(cropped, cropped, mask=mask)
	return dst

def predict(model, img, conf=0.5):
	results = model(img, conf=conf)
	if not results or len(results)==0:
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
			cv2.putText(img, name+':'+str(round(confidence,2)),(int(bbox[0]),int(bbox[1]-40)), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
			write_to_excel(name, img, current_time)
			text = is_target_object(name)
			if text != '':
				sound_notification(text)
	return (), img

def is_target_object(obj):
	text = ""
	if obj == 'nohandrailmidleft':
		text = 'mohon pegang handrail'
	elif obj == 'nohandrailleftfar':
		text = 'mohon pegang handrail'
	elif obj == 'nohandrailmidright':
		text = 'mohon pegang handrail'
	elif obj == 'nohandrailrightfar':
		text = 'mohon pegang handrail'
	elif obj == 'nohandrailupleft':
		text = 'mohon pegang handrail'
	elif obj == 'nohandraillowright':
		text = 'mohon pegang handrail'
	return text

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


#video_path = "0726.mp4"
#video_path = "tanggakiri.mp4"

model = YOLO('yolov8n.pt')

pts = np.array([[286,356],[236,250],[293,106],[416,76],[416,105],[326,135],[274,255],[311,346]])
#pts = np.array([[200,356],[150,250],[200,106],[300,76],[416,105],[326,135],[274,255],[311,346]])

cam_stream = CamStream(0)
cam_stream.start()

wb = Workbook()
ws = wb.active

frame_processed = 0
start = time.time()
while True:
	if cam_stream.stopped is True:
		break
	else:
		frame=cam_stream.read()
		frame = cv2.resize(frame, (416,416))
		#frame = crop_image(frame, pts)
		data, frame = predict(model, frame)
		frame_processed += 1
		cv2.imshow('cctv', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cam_stream.stop()
cv2.destroyAllWindows()

wb.save('handrail.xlsx')
datatodb.write_to_db()