import os
import numpy as np
import cv2
from ultralytics import YOLO
from threading import Thread, Lock
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

# Global sound lock dan timer
sound_lock = Lock()
last_sound_time = datetime.datetime.now()
MIN_SOUND_INTERVAL = 2  # minimal interval dalam detik antara suara

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

# Modify the OriginalStream class for better performance
class OriginalStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        self.vcap = cv2.VideoCapture(self.stream_id)
        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing stream.")
            exit(0)
        
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
            # Resize for display
            self.frame = cv2.resize(self.frame, (416, 416))
        self.vcap.release()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True

# Modify the start_detection function to handle streams separately
async def start_detection(cam_stream, model, processed_video, original_video, feature_pick):
    global frame_processed
    
    # Start original stream
    original_stream = OriginalStream(cam_stream.stream_id)
    original_stream.start()
    
    # Start original video update in a separate task
    original_task = asyncio.create_task(update_original_video(original_stream, original_video))
    
    try:
        if feature_pick == 'Handrail Detection':
            while True:
                if cam_stream.stopped:
                    break
                frame = cam_stream.read()
                frame = cv2.resize(frame, (416, 416))
                frame = await predict_handrail(model, frame, frame_processed)
                frame_processed += 1

                if frame_processed == 1:
                    processed_video.src = None
                
                processed_video.src_base64 = save_frame(frame)
                processed_video.update()

        elif feature_pick == 'Line of Fire Detection':
            while True:
                if cam_stream.stopped:
                    break
                frame = cam_stream.read()
                frame = cv2.resize(frame, (416, 416))
                frame = await predict_line_of_fire(model, frame, frame_processed)
                frame_processed += 1

                if frame_processed == 1:
                    processed_video.src = None
                
                processed_video.src_base64 = save_frame(frame)
                processed_video.update()

        elif feature_pick == 'Safety Equipment Detection':
            while True:
                if cam_stream.stopped:
                    break
                frame = cam_stream.read()
                frame = cv2.resize(frame, (416, 416))
                frame = await predict_safety_equipment(model, frame, frame_processed)
                frame_processed += 1

                if frame_processed == 1:
                    processed_video.src = None
                
                processed_video.src_base64 = save_frame(frame)
                processed_video.update()
    
    finally:
        # Cleanup
        original_stream.stop()
        cam_stream.stop()
        original_task.cancel()
        cv2.destroyAllWindows()

# Add a new function to update original video separately
async def update_original_video(original_stream, original_video):
    while not original_stream.stopped:
        frame = original_stream.read()
        original_video.src_base64 = save_frame(frame)
        original_video.update()
        await asyncio.sleep(0.001)  # Small delay to prevent overload

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
            print('Area not clear')
            # Ganti playsound dengan versi async
            play_sound_async('alerts/alert_lof.mp3')
            # Tidak perlu time.sleep lagi karena sound dijalankan di thread terpisah
            export_data.write_to_excel(ws, image_folder, 'Area not clear', img, current_time, frame_count)
        else:
            print('Area clear')

    return img

async def predict_safety_equipment(model, img, frame_count, conf=0.3):
    #crop_img = img.copy()
    #crop_pts = np.array([[450, 0], [640, 0], [640, 640], [450, 640]])
    #crop_img = crop(crop_img, crop_pts)
    await asyncio.sleep(0.001)
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

            if name == 'person':
                bbox_color = (0, 0, 255)
                is_no_helmet = True

            cv2.rectangle(img, (lx, ly), (ux, uy), bbox_color, 1)
            cv2.putText(img, name + ':' + str(round(confidence, 2)), (int(bbox[0]), int(bbox[1] - 40)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, bbox_color, 1)

        if is_no_helmet:
            print('No helmet')
            # Ganti playsound dengan versi async
            play_sound_async('alerts/alert_se.mp3')
            # Tidak perlu time.sleep lagi
            export_data.write_to_excel(ws, image_folder, 'No helmet', img, current_time, frame_count)

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
            print('not holding handrail')
            ser.write(b'on')
            # Ganti playsound dengan versi async
            play_sound_async('alerts/alert_hr.mp3')
            # Tidak perlu time.sleep lagi
            export_data.write_to_excel(ws, image_folder, name, img, current_time, frame_count)
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


class LoginView(View):
    def __init__(self, page: Page, on_login_success):
        super().__init__(route="/login")
        self.page = page
        self.on_login_success = on_login_success
        
    def build(self):
        self.username = TextField(
            label='Username',
            border=InputBorder.NONE,
            filled=True,
            prefix_icon=Icons.PERSON
        )
        self.password = TextField(
            label='Password',
            password=True,
            can_reveal_password=True,
            border=InputBorder.NONE,
            filled=True,
            prefix_icon=Icons.LOCK
        )
        
        return Container(
            bgcolor=colors.WHITE,
            content=Column(
                [
                    Container(
                        alignment=alignment.center,
                        margin=margin.only(bottom=16),
                        content=Column(
                            [
                                Image(src='images/pertamina.png', width=100),
                                Text(
                                    'A I LOPE U',
                                    style=TextStyle(weight=FontWeight.W_800, color=colors.BLACK, size=50)
                                ),
                            ],
                            alignment=CrossAxisAlignment.CENTER
                        )
                    ),
                    Container(
                        width=400,
                        padding=padding.all(10),
                        content=Column(
                            [
                                Text(
                                    'Please login to start the app',
                                    style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16),
                                    text_align=TextAlign.CENTER,  # Center the text
                                ),
                                Container(  # Wrap TextField in Container for width control
                                    width=300,
                                    content=self.username,
                                ),
                                Container(  # Wrap TextField in Container for width control
                                    width=300,
                                    content=self.password,
                                ),
                                Container(  # Wrap Button in Container for width control
                                    width=300,
                                    content=FilledButton(
                                        'Login',
                                        on_click=self.login,
                                        style=ButtonStyle(
                                            color=colors.WHITE,
                                            bgcolor=colors.RED,
                                            padding=padding.symmetric(vertical=20)
                                        )
                                    )
                                )
                            ],
                            horizontal_alignment=CrossAxisAlignment.CENTER,  # Center horizontally
                            alignment=MainAxisAlignment.CENTER,  # Center vertically
                            spacing=20
                        )
                    )
                ],
                horizontal_alignment=CrossAxisAlignment.CENTER,  # Center all content horizontally
                alignment=MainAxisAlignment.CENTER,  # Center all content vertically
            ),
            expand=True,
            alignment=alignment.center  # Center the main container
        )

    def login(self, e):
        if self.username.value == 'admin' and self.password.value == 'admin':
            self.on_login_success()
        else:
            self.page.dialog = AlertDialog(title=Text("Login failed. Please try again"))
            self.page.dialog.open = True
            self.password.value = ""
            self.password.update()
            self.page.update()

class MainView(View):
    def __init__(self, page: Page):
        super().__init__(route="/main")
        self.page = page
        self.is_running = False
        self.setup_controls()

    def setup_controls(self):
        self.result_video = Image(
            width=400,
            height=300,
            border_radius=border_radius.all(16),
            fit=ImageFit.CONTAIN,
            src_base64=None,
            src='images/black.png'
        )

        self.original_video = Image(
            width=400,
            height=300,
            border_radius=border_radius.all(16),
            fit=ImageFit.CONTAIN,
            src_base64=None,
            src='images/black.png'
        )

        self.feature_picker = Dropdown(
            value="Handrail Detection",
            alignment=alignment.center,
            width=500,
            border_color=colors.GREY,
            text_style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16),
            fill_color=colors.GREY_200,
            options=[
                dropdown.Option("Handrail Detection"),
                dropdown.Option("Line of Fire Detection"),
                dropdown.Option("Safety Equipment Detection"),
            ],
            autofocus=True
        )

        self.button = FilledButton(
            'Start',
            on_click=self.start_or_stop_app,
            style=ButtonStyle(
                color=colors.WHITE,
                bgcolor=colors.GREEN,
                padding=padding.symmetric(vertical=20)
            )
        )
        # Kemudian bungkus dalam container
        self.start_stop_button = Container(
            width=500,
            content=self.button
        )
        

    def build(self):
        return Container(
            bgcolor=colors.WHITE,
            content=Column(
                [
                    Container(
                        alignment=alignment.center,
                        margin=margin.only(bottom=16),
                        content=Column(
                            [
                                Image(src='images/pertamina.png', width=100),
                                Text(
                                    'A I LOPE U',
                                    style=TextStyle(weight=FontWeight.W_800, color=colors.BLACK, size=50)
                                ),
                            ],
                            horizontal_alignment=CrossAxisAlignment.CENTER,
                            alignment=MainAxisAlignment.CENTER,
                        )
                    ),
                    Row(
                        [
                            self.create_video_container("Deteksi", self.result_video),
                            self.create_video_container("Video Original", self.original_video),
                            self.create_control_container(),
                        ],
                        alignment=MainAxisAlignment.CENTER,
                        vertical_alignment=CrossAxisAlignment.START,
                        spacing=50,
                    ),
                ],
                horizontal_alignment=CrossAxisAlignment.CENTER,
                alignment=MainAxisAlignment.CENTER,
            ),
            padding=padding.all(10),
            expand=True,
            alignment=alignment.center,
        )

    def create_video_container(self, title, video):
        return Container(
            width=400,
            content=Column(
                [
                    Text(title, style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16)),
                    Container(
                        alignment=alignment.center,
                        border_radius=border_radius.all(20),
                        bgcolor=colors.BLACK,
                        width=400,
                        height=300,
                        content=video
                    ),
                ],
                spacing=16
            ),
        )

    def create_control_container(self):
        return Container(
            width=400,
            padding=padding.all(10),
            alignment=alignment.center,
            content=Column(
                [
                    Text(
                        'Choose what you want to detect',
                        style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16)
                    ),
                    self.feature_picker,
                    self.start_stop_button,
                    OutlinedButton(
                        'Export Data',
                        on_click=lambda e: export_data.export_to_excel(wb, image_folder, frame_processed),
                        width=500,
                        style=ButtonStyle(
                            color=colors.BLACK,
                            padding=padding.symmetric(vertical=20)
                        )
                    ),
                ],
            )
        )

    def start_or_stop_app(self, e):
        global detection_thread, cam_stream, model, ser
        if self.is_running:
            cam_stream.stop()
            self.button.text = "Start"
            self.button.style.bgcolor = colors.GREEN

            if detection_thread is not None:
                detection_thread.join()
                detection_thread = None
        else:
            if self.feature_picker.value == 'Handrail Detection':
                model = YOLO('models/handrail.pt')
                cam_stream = CamStream(video_path.video_path_handrail)
                ser = serial.Serial('COM5', 115200, timeout=1)
            elif self.feature_picker.value == 'Line of Fire Detection':
                model = YOLO('models/line_of_fire.pt')
                cam_stream = CamStream(video_path.video_path_line_of_fire)
            elif self.feature_picker.value == 'Safety Equipment Detection':
                model = YOLO('models/yolo11n.pt')
                cam_stream = CamStream(video_path.video_path_safety_equipment)

            cam_stream.start()
            self.button.text = "Stop"
            self.button.style.bgcolor = colors.RED
            self.page.update()

            detection_thread = Thread(
                target=asyncio.run,
                args=(start_detection(cam_stream, model, self.result_video, self.original_video, self.feature_picker.value),)
            )
            detection_thread.start()

        self.start_stop_button.update()
        self.is_running = not self.is_running

def main(page: Page):
    page.title = 'A I LOPE U'
    page.padding = 0  # Remove default padding
    page.spacing = 0  # Remove default spacing
    page.bgcolor = colors.WHITE
    page.window_bgcolor = colors.WHITE  # Set window background color
    page.theme_mode = ThemeMode.LIGHT
    
    def route_change(route):
        page.views.clear()
        if page.route == "/login":
            page.views.append(LoginView(page, lambda: page.go("/main")))
        elif page.route == "/main":
            page.views.append(MainView(page))
        page.update()

    page.on_route_change = route_change
    page.go("/login")

frame_processed = 0
video_path_handrail = video_path.video_path_handrail
video_path_line_of_fire = video_path.video_path_line_of_fire
video_path_safety_equipment = video_path.video_path_safety_equipment
esp32_ip = "http://192.168.100.163/send-data"

if __name__ == "__main__":
    # Initialize global variables
    frame_processed = 0
    detection_thread = None
    model = None
    cam_stream = None
    wb = Workbook()
    ws = wb.active
    
    image_folder = "temp_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    app(main)
    
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)