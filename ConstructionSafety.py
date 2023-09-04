from ultralytics import YOLO
import cvzone
import cv2

# Getting video source.
# video = cv2.VideoCapture("../static/videos/video-1.mp4")
video = cv2.VideoCapture("E:/Syed Muhammad Zaid/My Projects/ConstructionSafetyMonitor/static/videos/video-1.mp4")

# Creating YOLO MODEL.
model = YOLO(
    "../Weights/TBMConstructionSafety.pt")  # Trained By Me; ConstructionSafety Model (Based on YOLO version 8 - Nano)

# Classnames used in the model training.
classnames = ['Excavator', 'Gloves', 'Construction-Hat', 'Ladder', 'Mask', 'NO-Construction-Hat', 'NO-Mask',
              'NO-Safety-Vest', 'Person', 'SUV', 'Safety-Cone', 'Safety-Vest', 'bus', 'Dump-Truck', 'Fire-Hydrant',
              'Machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle',
              'wheel loader']

# Looping through the video and getting the detection started.
while True:
    success, img = video.read()
    results = model(img, stream=True)  # Creating results to display the video.

    for i in results:
        boxes = i.boxes
        for box in boxes:
            # Creating bounding box and setting up axes of image.
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculating width and height from subtracting the width and height.
            width, height = x2 - x1, y2 - y1

            bbox = x1, y1, width, height

            cv2.putText(img, "CONSTRUCTION SAFETY DETECTION", org=(50, 50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2, color=(0, 0, 0), thickness=3)

            # Calculating confidence level.
            confidence = round(box.conf[0].item(), 2)

            # Creating the corner rectangle from cvzone library.
            cvzone.cornerRect(img, bbox)

            # Below variable holds the class ID number. (e.g. 1.00)
            clsId = box.cls[0]
            clsId = int(clsId)

            # Putting text over the rectangle.
            cvzone.putTextRect(img, f'{confidence} {classnames[clsId]}', (max(0, x1), max(20, y1)), scale=1,
                               thickness=1, font=cv2.FONT_HERSHEY_PLAIN)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


#Hell
