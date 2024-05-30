import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load your YOLO model
model = YOLO('best.pt')


# Function for mouse callback to print RGB values
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


# Create a window and set a mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Read the image
frame = cv2.imread('Resources/plate.png')
if frame is None:
    print("Failed to load image.")
else:
    # Resize the image for processing
    frame = cv2.resize(frame, (1020, 500))

    # Perform YOLO model prediction on the image
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Load class list from file
    with open("coco1.txt", "r") as my_file:
        data = my_file.read()
    class_list = data.split("\n")

    # Define the area of interest (if needed)
    area = [(34, 400), (16, 456), (1015, 451), (978, 387)]

    # Initialize a set to keep track of processed numbers
    processed_numbers = set()

    # Open file for writing car plate data
    with open("car_plate_data.txt", "a") as file:
        file.write("NumberPlate \tDate \tTime \n")  # Writing column headers

    # Iterate over detection results
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2

        # Draw the rectangle on the image
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Optionally, perform OCR to extract text from the detected area
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 10, 20, 20)
        text = pytesseract.image_to_string(gray).strip()
        text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')
        print(text)
        if text not in processed_numbers:
            processed_numbers.add(text)
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("car_plate_data.txt", "a") as file:
                file.write(f"{text}\t{current_datetime}\n")

    # Show the processed image
    cv2.imshow("RGB", frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()
