import os
import cv2
from ultralytics import YOLO
from threading import Thread
from openpyxl.drawing.image import Image
from openpyxl import Workbook
import shutil
from flet import *
import asyncio
import requests
import export_data
import serial
import camstream
import detection

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
        src='sample_image.png'
    )

    # Dialogs
    connect_success = AlertDialog(title=Text("Device is connected"))
    login_success = AlertDialog(title=Text("Logged In"))
    login_failed = AlertDialog(title=Text("Login failed. Please try again"))


    # Event Handlers
    def start_or_stop_app(e):
        global detection_thread
        nonlocal is_running
        nonlocal is_loading
        if is_running:
            # Change to "Start" state
            cam_stream.stop()
            start_stop_button_ref.current.text = "Start"
            start_stop_button_ref.current.style.bgcolor = colors.GREEN

            if detection_thread is not None:
                detection_thread.join()  # Ensure thread completes before moving on
                detection_thread = None
        else:
            # Start main.py and video updates
            cam_stream.start()
            start_stop_button_ref.current.text = "Stop"
            start_stop_button_ref.current.style.bgcolor = colors.RED
            # is_loading = True
            # loading_spinner.visible = True
            page.update()

            detection_thread = Thread(target=asyncio.run, args=(detection.start_detection(model, result_video, cam_stream, ws, image_folder, frame_processed),))
            detection_thread.start()

            # is_loading = False
            # loading_spinner.visible = False

        start_stop_button_ref.current.update()
        page.update()
        is_running = not is_running

    async def feedback_test(e):
        # WIFI
        response = requests.post(esp32_ip, data='on')
        # Serial USB
        # ser.write(b'on')
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
                                Image(src='pertamina.png', width=100),
                                Text('PERTAMINA \n"AIDS"', style=TextStyle(weight=FontWeight.W_800, color=colors.BLACK, size=50)),
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
video_path = "rtsp://admin:pertamina321@10.205.64.111:554/Streaming/Channels/301"
esp32_ip = "http://192.168.100.176/send-data"

model = YOLO('yolov8n.pt')
cam_stream = camstream.CamStream('test.mp4')

wb = Workbook()
ws = wb.active

image_folder = "temp_images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# ser = serial.Serial('COM5', 115200, timeout=1)

if __name__ == "__main__":
    app(main)
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
