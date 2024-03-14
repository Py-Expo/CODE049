import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("C:\\Users\\prith\\Desktop\\CODE049\\runs\\detect\\train3\\weights\\best.pt")  # Replace with the path to your trained model weights

# Open the video file
video_path = 'C:\\Users\\prith\\Desktop\\CODE049\\parking_lot_1.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a video writer object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame (optional)
    cv2.imshow('Object Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()