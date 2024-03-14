from ultralytics import YOLO

# Load the YOLOv8 model architecture
model = YOLO('yolov8n.yaml')  # or 'yolov8s.yaml', 'yolov8m.yaml', 'yolov8l.yaml'

# Set the path to your dataset configuration file
dataset_path = 'data.yaml'

# Train the model
model.train(data=dataset_path, epochs=100, imgsz=640)

# Evaluate the model
metrics = model.val(data=dataset_path)