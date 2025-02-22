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
import asyncio
import core.export_data as export_data
import serial
# Create a new file called video_path.py and add your video/cctv rtsp url
import video_path
#core dan komponen terpisah
from components.system_monitor import SystemMonitor
from components.login import LoginView
from core.videostream import OriginalStream, CamStream
from core.utils import crop, calculate_iou, calculate_red_pixel_std, save_frame, play_sound_async

async def start_detection(cam_stream, model, processed_video, feature_pick):
    global frame_processed
    frame_processed = 0  # Reset counter when starting detection
    
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
        cam_stream.stop()
        cv2.destroyAllWindows()

# Add a new function to update original video separately
async def update_original_video(original_stream, original_video):
    while not original_stream.stopped:
        frame = original_stream.read()
        original_video.src_base64 = save_frame(frame)
        original_video.update()
        await asyncio.sleep(0.001)  # Small delay to prevent overload

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

class MainView(View):
    def __init__(self, page: Page):
        super().__init__(route="/main")
        self.page = page
        self.is_detection_running = False
        self.is_detection2_running = False
        self.is_original_running = False
        self.is_original2_running = False
        self.setup_controls()
        self.system_monitor = SystemMonitor()

    def setup_controls(self):
        # First set of videos
        self.result_video = Image(
            width=300,  # Updated width
            height=225,  # Updated height
            border_radius=border_radius.all(16),
            fit=ImageFit.CONTAIN,
            src_base64=None,
            src='images/black.png'
        )

        self.original_video = Image(
            width=300,  # Updated width
            height=225,  # Updated height
            border_radius=border_radius.all(16),
            fit=ImageFit.CONTAIN,
            src_base64=None,
            src='images/black.png'
        )

        self.result_video2 = Image(
            width=300,  # Updated width
            height=225,  # Updated height
            border_radius=border_radius.all(16),
            fit=ImageFit.CONTAIN,
            src_base64=None,
            src='images/black.png'
        )

        self.original_video2 = Image(
            width=300,  # Updated width
            height=225,  # Updated height
            border_radius=border_radius.all(16),
            fit=ImageFit.CONTAIN,
            src_base64=None,
            src='images/black.png'
        )

        # First set of buttons
        self.detection_button = FilledButton(
            'Start Detection',
            on_click=self.toggle_detection,
            style=ButtonStyle(
                color=colors.WHITE,
                bgcolor=colors.GREEN,
                padding=padding.symmetric(vertical=20)
            )
        )

        self.original_button = FilledButton(
            'Start Original',
            on_click=self.toggle_original,
            style=ButtonStyle(
                color=colors.WHITE,
                bgcolor=colors.GREEN,
                padding=padding.symmetric(vertical=20)
            )
        )

        # Second set of buttons
        self.detection_button2 = FilledButton(
            'Start Detection 2',
            on_click=self.toggle_detection2,
            style=ButtonStyle(
                color=colors.WHITE,
                bgcolor=colors.GREEN,
                padding=padding.symmetric(vertical=20)
            ),
            visible=False
        )

        self.original_button2 = FilledButton(
            'Start Original 2',
            on_click=self.toggle_original2,
            style=ButtonStyle(
                color=colors.WHITE,
                bgcolor=colors.GREEN,
                padding=padding.symmetric(vertical=20)
            ),
            visible=False
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
            on_change=self.handle_dropdown_change,
            autofocus=True
        )

        # Container for second set of videos
        self.second_video_row = Row(
            [
                self.create_video_container("Deteksi 2", self.result_video2, self.detection_button2),
                self.create_video_container("Video Original 2", self.original_video2, self.original_button2),
            ],
            alignment=MainAxisAlignment.CENTER,
            vertical_alignment=CrossAxisAlignment.START,
            spacing=50,
            visible=False
        )

    def create_video_container(self, title, video, control_button, width=400):
        return Container(
            width=width,
            visible=True,  # Will be controlled by handle_dropdown_change
            content=Column(
                [
                    Text(title, style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16)),
                    Container(
                        alignment=alignment.center,
                        border_radius=border_radius.all(20),
                        bgcolor=colors.BLACK,
                        width=width,
                        height=225,  # Adjusted height to maintain aspect ratio
                        content=video
                    ),
                    Container(
                        width=width,
                        content=control_button
                    )
                ],
                spacing=16,
                horizontal_alignment=CrossAxisAlignment.CENTER,
            ),
        )

    def build(self):
        self.video_container1 = self.create_video_container(
            "Deteksi", 
            self.result_video, 
            self.detection_button
        )
        self.video_container2 = self.create_video_container(
            "Video Original", 
            self.original_video, 
            self.original_button
        )
        self.video_container3 = self.create_video_container(
            "Deteksi 2", 
            self.result_video2, 
            self.detection_button2,
            width=300
        )
        self.video_container4 = self.create_video_container(
            "Video Original 2", 
            self.original_video2, 
            self.original_button2,
            width=300
        )
        
        # Store containers as instance variables for easy access
        self.video_containers = [
            self.video_container1,
            self.video_container2,
            self.video_container3,
            self.video_container4
        ]

        # Initially hide second set of videos completely
        self.video_container3.visible = False
        self.video_container4.visible = False
        
        all_videos_row = Row(
            self.video_containers,
            alignment=MainAxisAlignment.CENTER,
            vertical_alignment=CrossAxisAlignment.START,
            spacing=20,
        )

        controls_row = Container(
            content=Column(
                [
                    Container(
                        content=Column(
                            [
                                Text(
                                    'Choose what you want to detect',
                                    style=TextStyle(weight=FontWeight.W_600, color=colors.BLACK, size=16),
                                    text_align=TextAlign.CENTER,
                                ),
                                self.feature_picker,
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
                            spacing=10,
                            horizontal_alignment=CrossAxisAlignment.CENTER,
                        ),
                        padding=padding.symmetric(vertical=20),
                    ),
                ],
                horizontal_alignment=CrossAxisAlignment.CENTER,
            ),
            alignment=alignment.center,
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
                            horizontal_alignment=CrossAxisAlignment.CENTER,
                            alignment=MainAxisAlignment.CENTER,
                        )
                    ),
                    all_videos_row,
                    controls_row,
                    self.system_monitor,
                ],
                horizontal_alignment=CrossAxisAlignment.CENTER,
                alignment=MainAxisAlignment.CENTER,
                spacing=20,
            ),
            padding=padding.all(10),
            expand=True,
            alignment=alignment.center,
        )

    def handle_dropdown_change(self, e):
        # Stop all videos if running
        if self.is_detection_running:
            self.stop_detection()
        if self.is_detection2_running:
            self.stop_detection2()
        if self.is_original_running:
            self.stop_original()
        if self.is_original2_running:
            self.stop_original2()

        is_safety_equipment = e.data == "Safety Equipment Detection"
        
        if is_safety_equipment:
            # Show all 4 videos with smaller width
            self.video_container1.width = 300
            self.video_container2.width = 300
            self.video_container3.visible = True
            self.video_container4.visible = True
            self.result_video.width = 300
            self.original_video.width = 300
            self.result_video2.width = 300
            self.original_video2.width = 300
        else:
            # Show only 2 videos with larger width
            self.video_container1.width = 400
            self.video_container2.width = 400
            self.video_container3.visible = False
            self.video_container4.visible = False
            self.result_video.width = 400
            self.original_video.width = 400
        
        # Update video and button visibility
        self.result_video2.visible = is_safety_equipment
        self.original_video2.visible = is_safety_equipment
        self.detection_button2.visible = is_safety_equipment
        self.original_button2.visible = is_safety_equipment
        
        self.page.update()

    def stop_detection(self):
        global detection_thread, cam_stream
        if cam_stream:
            cam_stream.stop()
        
        self.detection_button.text = "Start Detection"
        self.detection_button.style.bgcolor = colors.GREEN
        
        if detection_thread is not None:
            detection_thread.join()
            detection_thread = None
        
        self.result_video.src_base64 = None
        self.result_video.src = 'images/black.png'
        self.result_video.update()
        self.is_detection_running = False
        self.detection_button.update()

    def stop_detection2(self):
        global detection_thread2, cam_stream2
        if cam_stream2:
            cam_stream2.stop()
        
        self.detection_button2.text = "Start Detection 2"
        self.detection_button2.style.bgcolor = colors.GREEN
        
        if detection_thread2 is not None:
            detection_thread2.join()
            detection_thread2 = None
        
        self.result_video2.src_base64 = None
        self.result_video2.src = 'images/black.png'
        self.result_video2.update()
        self.is_detection2_running = False
        self.detection_button2.update()

    def stop_original(self):
        if hasattr(self, 'original_stream'):
            self.original_stream.stop()
        self.original_button.text = "Start Original"
        self.original_button.style.bgcolor = colors.GREEN
        self.original_video.src = 'images/black.png'
        self.original_video.src_base64 = None
        self.original_video.update()
        self.is_original_running = False
        self.original_button.update()

    def stop_original2(self):
        if hasattr(self, 'original_stream2'):
            self.original_stream2.stop()
        self.original_button2.text = "Start Original 2"
        self.original_button2.style.bgcolor = colors.GREEN
        self.original_video2.src = 'images/black.png'
        self.original_video2.src_base64 = None
        self.original_video2.update()
        self.is_original2_running = False
        self.original_button2.update()

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
    
    def toggle_detection(self, e):
        global detection_thread, cam_stream, model, ser
        if self.is_detection_running:
            self.stop_detection()
        else:
            # Start new detection
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
            self.detection_button.text = "Stop Detection"
            self.detection_button.style.bgcolor = colors.RED
            self.page.update()

            detection_thread = Thread(
                target=asyncio.run,
                args=(start_detection(cam_stream, model, self.result_video, self.feature_picker.value),)
            )
            detection_thread.start()
            self.is_detection_running = True
            self.detection_button.update()

    def toggle_detection2(self, e):
        global detection_thread2, cam_stream2, model2
        if self.is_detection2_running:
            self.stop_detection2()
        else:
            # Start new detection (always Safety Equipment)
            model2 = YOLO('models/yolo11n.pt')
            cam_stream2 = CamStream(video_path.video_path_safety_equipment2)  # New video path

            cam_stream2.start()
            self.detection_button2.text = "Stop Detection 2"
            self.detection_button2.style.bgcolor = colors.RED
            self.page.update()

            detection_thread2 = Thread(
                target=asyncio.run,
                args=(start_detection(cam_stream2, model2, self.result_video2, self.feature_picker.value),)
            )
            detection_thread2.start()
            self.is_detection2_running = True
            self.detection_button2.update()

    def toggle_original(self, e):
        if self.is_original_running:
            self.stop_original()
        else:
            # Select video path based on feature selection
            if self.feature_picker.value == 'Handrail Detection':
                video_path_selected = video_path.video_path_handrail
            elif self.feature_picker.value == 'Line of Fire Detection':
                video_path_selected = video_path.video_path_line_of_fire
            elif self.feature_picker.value == 'Safety Equipment Detection':
                video_path_selected = video_path.video_path_safety_equipment
            
            self.original_stream = OriginalStream(video_path_selected)
            self.original_stream.start()
            self.original_button.text = "Stop Original"
            self.original_button.style.bgcolor = colors.RED
            
            async def update_original():
                while not self.original_stream.stopped:
                    frame = self.original_stream.read()
                    self.original_video.src_base64 = save_frame(frame)
                    self.original_video.update()
                    await asyncio.sleep(0.001)

            Thread(target=asyncio.run, args=(update_original(),)).start()
            self.is_original_running = True
            self.original_button.update()

    def toggle_original2(self, e):
        if self.is_original2_running:
            self.stop_original2()
        else:
            # Always use second safety equipment video path
            video_path_selected = video_path.video_path_safety_equipment2  # New video path
            
            self.original_stream2 = OriginalStream(video_path_selected)
            self.original_stream2.start()
            self.original_button2.text = "Stop Original 2"
            self.original_button2.style.bgcolor = colors.RED
            
            async def update_original2():
                while not self.original_stream2.stopped:
                    frame = self.original_stream2.read()
                    self.original_video2.src_base64 = save_frame(frame)
                    self.original_video2.update()
                    await asyncio.sleep(0.001)

            Thread(target=asyncio.run, args=(update_original2(),)).start()
            self.is_original2_running = True
            self.original_button2.update()

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