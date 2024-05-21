from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    # Plot results on the frame
    frame_ = results[0].plot()

    cv2.imshow("frame", frame_)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

    for result in results:
        for box in result.boxes:
            print(type(box))
            print(box)
