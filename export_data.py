import cv2
from openpyxl.drawing.image import Image
import shutil
import datetime

def write_to_excel(ws, image_folder, data, img, current_time, frame_count):
    img_filename = f"{image_folder}/frame_image_{frame_count}.png"
    cv2.imwrite(img_filename, img)
    img = Image(img_filename)

    ws.append([data, current_time])
    ws.add_image(img, 'C' + str(ws.max_row))

    adjust_dimensions(ws)

def adjust_dimensions(ws):
    for col in ws.columns:
        max_length = max(len(str(cell.value)) for cell in col if cell.value)
        column = col[0].column_letter
        ws.column_dimensions[column].width = max_length + 2

    for row in ws.iter_rows():
        for cell in row:
            ws.row_dimensions[cell.row].height = 300

def export_to_excel(wb, image_folder, frame_processed):
    today = datetime.date.today()
    if frame_processed > 0:  # Check if there are frames to export
        wb.save(f'{today}.xlsx')  # Save the workbook
        print('Data Exported')
        shutil.rmtree(image_folder)  # Clean up temporary images