import os
import numpy as np
import cv2
from ultralytics import YOLO
from threading import Thread
import datetime
from openpyxl.drawing.image import Image
from openpyxl import Workbook
import shutil
from flet import *
import base64
import asyncio
import requests
import export_data
import serial
import time
from playsound import playsound
# Create a new file called video_path.py and add your video/cctv rtsp url
import video_path

# CamStream class for video stream handling
class CamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        self.vcap = cv2.VideoCapture(self.stream_id)
        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing stream.")
            exit(0)

        fps_input_stream = int(self.vcap.get(5))
        print(f"FPS of hardware/input stream: {fps_input_stream}")

        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.vcap.read()
            if not self.grabbed:
                self.stopped = True
                break
        self.vcap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

async def predict_line_of_fire(model, img, frame_count, conf=0.3):
    results = model(img, conf=conf, verbose=False)
    if not results:
        return img

    current_time = datetime.datetime.now().strftime("%I:%M%p")

    tractor_coords = []
    person_coords = []
    
    for result in results:
        count = result.boxes.shape[0]
        for i in range(count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            bbox_color = (255, 0, 255)
            if name in {'person', 'tracktor', 'helmet'}:
                confidence = float(result.boxes.conf[i].item())
                bbox = result.boxes.xywh[i].cpu().numpy()
                x, y, w, h = bbox
                lx, ux = int(x - w / 2), int(x + w / 2)
                ly, uy = int(y - h / 2), int(y + h / 2)

                if name == 'tracktor':
                    print(lx)
                    lx, ux, ly, uy = lx-10, ux+10, ly-10, uy+10
                    tractor_coords = [lx, ux, ly, uy]
                    bbox_color = (0, 0, 255)
                    print(lx)
                elif name == 'person':
                    person_coords.append([lx, ux, ly, uy])

                cv2.rectangle(img, (lx, ly), (ux, uy), bbox_color, 1)
                cv2.putText(img, name + ':' + str(round(confidence, 2)), (int(bbox[0]), int(bbox[1] - 40)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, bbox_color, 1)

    if tractor_coords and person_coords:
        print(tractor_coords)
        print(person_coords)
        iou_values = [calculate_iou(tractor_coords, person) for person in person_coords]
        print(iou_values)
        if any(iou > 0 for iou in iou_values):
            # test
            print('Area not clear')
            playsound('alerts/alert_lof.mp3')
            # Wifi
            # response = requests.post(esp32_ip, data='on')
            # Serial USB
            # ser.write(b'on')
            export_data.write_to_excel(ws, image_folder, 'Area not clear', img, current_time, frame_count)
            time.sleep(1)
        else:
            print('Area clear')

    return img

async def predict_safety_equipment(model, img, frame_count, conf=0.3):
    #crop_img = img.copy()
    #crop_pts = np.array([[450, 0], [640, 0], [640, 640], [450, 640]])
    #crop_img = crop(crop_img, crop_pts)
    results = model(img, conf=conf, verbose=False)
    if not results:
        return img

    current_time = datetime.datetime.now().strftime("%I:%M%p")
    
    is_no_helmet = False
    for result in results:
        count = result.boxes.shape[0]
        for i in range(count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            bbox_color = (255, 0, 255)
            confidence = float(result.boxes.conf[i].item())
            bbox = result.boxes.xywh[i].cpu().numpy()
            x, y, w, h = bbox
            lx, ux = int(x - w / 2), int(x + w / 2)
            ly, uy = int(y - h / 2), int(y + h / 2)

            if name == 'no helmet':
                bbox_color = (0, 0, 255)
                is_no_helmet = True

            cv2.rectangle(img, (lx, ly), (ux, uy), bbox_color, 1)
            cv2.putText(img, name + ':' + str(round(confidence, 2)), (int(bbox[0]), int(bbox[1] - 40)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, bbox_color, 1)

    if is_no_helmet:
        print('No helmet')
        playsound('alerts/alert_se.mp3')
        # Wifi
        # response = requests.post(esp32_ip, data='on')
        # Serial USB
        # ser.write(b'on')
        export_data.write_to_excel(ws, image_folder, 'No helmet', img, current_time, frame_count)
        time.sleep(1)

    return img

async def predict_handrail(model, img, frame_count, conf=0.3):
    crop_img = img.copy()
    results = model(img, conf=conf, verbose=False)
    if not results:
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

                    # Check if the person is holding the handrail
                    if pred_px >= 20:
                        is_send = True
            else:
                return img

    if is_send:
        # test
        print('not holding handrail')
        # Wifi
        # response = requests.post(esp32_ip, data='on')
        # Serial USB
        # ser.write(b'on')
        playsound('alerts/alert_hr.mp3')
        export_data.write_to_excel(ws, image_folder, name, img, current_time, frame_count)
        time.sleep(1)
    else:
        print('handrail')

    return img

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

async def start_detection(cam_stream, model, video, feature_pick):
    global frame_b64
    global frame_processed
    if feature_pick == 'Handrail Detection':
        while True:
            if cam_stream.stopped:
                break
            frame = cam_stream.read()
            frame = cv2.resize(frame, (416, 416))
            frame = await predict_handrail(model, frame, frame_processed)
            frame_processed += 1

            if frame_processed == 1:
                video.src = None
            video.src_base64 = save_frame(frame)
            video.update()
    if feature_pick == 'Line of Fire Detection':
        while True:
            if cam_stream.stopped:
                break
            frame = cam_stream.read()
            frame = cv2.resize(frame, (416, 416))
            frame = await predict_line_of_fire(model, frame, frame_processed)
            frame_processed += 1

            if frame_processed == 1:
                video.src = None
            video.src_base64 = save_frame(frame)
            video.update()
    if feature_pick == 'Safety Equipment Detection':
        while True:
            if cam_stream.stopped:
                break
            frame = cam_stream.read()
            frame = cv2.resize(frame, (416, 416))
            frame = await predict_safety_equipment(model, frame, frame_processed)
            frame_processed += 1

            if frame_processed == 1:
                video.src = None
            video.src_base64 = save_frame(frame)
            video.update()
    
    cam_stream.stop()
    cv2.destroyAllWindows()

def main(page: Page):
    page.title = 'AIDS'
    page.vertical_alignment = MainAxisAlignment.CENTER
    page.bgcolor = colors.WHITE

    # Track button states
    is_running = False  # Track if the app is running
    is_loading = False  # Track if loading spinner should be shown
    start_stop_button_ref = Ref[FilledButton]()

    # Light indicator container
    indicator = Container(
        width=25,
        height=25,
        bgcolor=colors.RED,  # Start with red
        border_radius=25,
    )

    # Username and password fields
    username = TextField(label='Username', border=InputBorder.NONE, filled=True, prefix_icon=icons.PERSON)
    password = TextField(label='Password', password=True, can_reveal_password=True, border=InputBorder.NONE, filled=True, prefix_icon=icons.LOCK)

    # Loading spinner (only shown when is_loading=True)
    loading_spinner = Row(
        [
            ProgressRing(stroke_width=2, color=colors.BLUE_500),
            Text('Loading...')
        ],
        visible=False
    )

    result_video = Image(
        width=400,
        height=300,
        border_radius=border_radius.all(16),
        fit=ImageFit.CONTAIN,
        src_base64=None,
        src='images/black.png'
    )

    # Dialogs
    connect_success = AlertDialog(title=Text("Device is connected"))
    login_success = AlertDialog(title=Text("Logged In"))
    login_failed = AlertDialog(title=Text("Login failed. Please try again"))

    # Dropdown
    feature_picker = Dropdown(
        value="Handrail Detection",
        alignment=alignment.center,
        width=500,
        border_color=colors.BLACK,
        text_style=TextStyle(weight=FontWeight.W_600, color=colors.WHITE, size=16),
        fill_color=colors.BLACK,
        options=[
            dropdown.Option("Handrail Detection"),
            dropdown.Option("Line of Fire Detection"),
            dropdown.Option("Safety Equipment Detection"),
        ],
        autofocus=True
    )


    # Event Handlers
    def start_or_stop_app(e):
        global detection_thread
        nonlocal is_running
        nonlocal is_loading
        global ser
        global cam_stream
        global model
        if is_running:
            # Change to "Start" state
            cam_stream.stop()
            start_stop_button_ref.current.text = "Start"
            start_stop_button_ref.current.style.bgcolor = colors.GREEN

            if detection_thread is not None:
                detection_thread.join()  # Ensure thread completes before moving on
                detection_thread = None
        else:
            # Read feature_picker and load models and stuffs
            if feature_picker.value == 'Handrail Detection':
                model = YOLO('models/handrail.pt')
                cam_stream = CamStream(video_path_handrail)
                # ser = serial.Serial('COM5', 115200, timeout=1)
            elif feature_picker.value == 'Line of Fire Detection':
                model = YOLO('models/line_of_fire.pt')
                cam_stream = CamStream(video_path_line_of_fire)
            elif feature_picker.value == 'Safety Equipment Detection':
                model = YOLO('models/safety_equipment.pt')
                cam_stream = CamStream(video_path_safety_equipment)

            # Start main.py and video updates
            cam_stream.start()
            start_stop_button_ref.current.text = "Stop"
            start_stop_button_ref.current.style.bgcolor = colors.RED
            # is_loading = True
            # loading_spinner.visible = True
            page.update()

            detection_thread = Thread(target=asyncio.run, args=(start_detection(cam_stream, model, result_video, feature_picker.value),))
            detection_thread.start()

            # is_loading = False
            # loading_spinner.visible = False

        start_stop_button_ref.current.update()
        page.update()
        is_running = not is_running

    async def feedback_test(e):
        # response = requests.post(esp32_ip, data='on')
        ser.write(b'on')
        page.open(connect_success)
        indicator.bgcolor = colors.GREEN  # Change indicator color to green
        indicator.update()
        page.update()

    def login(e):
        if username.value == 'admin' and password.value == '123poleng':
            page.open(login_success)
            start_stop_button_ref.current.disabled = False  # Enable the Start/Stop button
            start_stop_button_ref.current.update()
        else:
            page.open(login_failed)
            password.value = ""
            password.update()

        page.update()

    # Layout setup
    page.add(
        Container(
            content=Row(
                [
                    # Left side: Login section
                    Container(
                        width=400,
                        content=Column(
                            [
                                Container(
                                    margin=margin.only(bottom=16),
                                    content=Column(
                                        [
                                            Image(src='images/pertamina.png', width=100),
                                            Text('PERTAMINA \n"AIDS"', style=TextStyle(weight=FontWeight.W_800, color=colors.BLACK, size=50)),
                                        ]
                                    )
                                ),
                                Container(
                                    margin=margin.only(bottom=16),
                                    content=Column(
                                        [
                                            username,
                                            password,
                                            FilledButton(
                                                'Login',
                                                on_click=login,
                                                adaptive=True,
                                                width=500,
                                                style=ButtonStyle(
                                                    color=colors.WHITE,
                                                    bgcolor=colors.RED,
                                                    padding=padding.symmetric(vertical=20)
                                                )
                                            ),
                                        ]
                                    )
                                ),
                                Text('Choose what you want to detect', style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16)),
                                feature_picker,
                                FilledButton(
                                    'Start',
                                    on_click=start_or_stop_app,
                                    adaptive=True,
                                    ref=start_stop_button_ref,
                                    width=500,
                                    disabled=True,  # Initially disabled
                                    style=ButtonStyle(
                                        color=colors.WHITE,
                                        bgcolor=colors.GREEN,
                                        padding=padding.symmetric(vertical=20)
                                    )
                                ),
                                OutlinedButton(
                                    'Export Data',
                                    on_click=lambda e:export_data.export_to_excel(wb, image_folder, frame_processed),
                                    adaptive=True,
                                    width=500,
                                    style=ButtonStyle(
                                        color=colors.BLACK,
                                        padding=padding.symmetric(vertical=20)
                                    )
                                ),
                            ],
                            spacing=16
                        ),
                    ),
                    # Right side: Video frame and feedback
                    Column(
                        [
                            Container(
                                alignment=alignment.center,
                                border_radius=border_radius.all(20),
                                bgcolor=colors.BLACK,
                                width=400,
                                height=300,
                                content=result_video
                            ),
                            Row(
                                [
                                    OutlinedButton(
                                        'Feedback Test',
                                        on_long_press=feedback_test,
                                        adaptive=True,
                                        width=150,
                                        style=ButtonStyle(
                                            color=colors.BLACK,
                                            padding=padding.symmetric(vertical=20)
                                        )
                                    ),
                                    indicator  # Light indicator container
                                ],
                                alignment=MainAxisAlignment.SPACE_BETWEEN,
                                width=400
                            ),
                            loading_spinner  # Show loading spinner if is_loading=True
                        ],
                        spacing=16
                    ),
                ],
                alignment=MainAxisAlignment.CENTER,
                vertical_alignment=CrossAxisAlignment.CENTER,
                spacing=50,
            ),
            padding=padding.all(10),
            alignment=alignment.center
        )
    )

frame_processed = 0
video_path_handrail = video_path.video_path_handrail
video_path_line_of_fire = video_path.video_path_line_of_fire
video_path_safety_equipment = video_path.video_path_safety_equipment
esp32_ip = "http://192.168.100.163/send-data"

model = None
cam_stream = None

wb = Workbook()
ws = wb.active

image_folder = "temp_images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

global ser

if __name__ == "__main__":
    app(main)
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)