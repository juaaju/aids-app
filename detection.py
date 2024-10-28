import cv2
import base64
import predict

def save_frame(frame):
    _, im_arr = cv2.imencode('.jpg', frame)
    im_b64 = base64.b64encode(im_arr)
    return im_b64.decode('utf-8')

async def start_detection(model, video, cam_stream, ws, image_folder, frame_processed, serial=''):
    while True:
        if cam_stream.stopped:
            break
        frame = cam_stream.read()
        frame = cv2.resize(frame, (416, 416))
        frame = await predict.predict(model, frame, frame_processed, ws, image_folder, serial=serial)
        frame_processed += 1

        if frame_processed == 1:
            video.src = None
        video.src_base64 = save_frame(frame)
        video.update()
    
    cam_stream.stop()
    cv2.destroyAllWindows()