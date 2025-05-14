import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\vivek\OneDrive\Desktop\results\my_modelZZZ\my_model.pt")  # Replace with the actual path of your trained model

# Define class names (Ensure this matches your dataset's YAML file)
class_names = {
    0: "Speed Limit 20 km/h", 
    1: "Speed Limit 30 km/h", 
    2: "Speed Limit 50 km/h", 
    3: "Speed Limit 60 km/h", 
    4: "Speed Limit 70 km/h", 
    5: "Speed Limit 80 km/h", 
    6: "End of Speed Limit 80 km/h", 
    7: "Speed Limit 100 km/h", 
    8: "Speed Limit 120 km/h", 
    9: "No Overtaking", 
    10: "No Overtaking (Trucks)", 
    11: "Right-of-way at intersection", 
    12: "Priority Road", 
    13: "Yield", 
    14: "Stop", 
    15: "No Vehicles", 
    16: "No Trucks", 
    17: "No Entry", 
    18: "General Caution", 
    19: "Dangerous Curve Left", 
    20: "Dangerous Curve Right", 
    21: "Double Curve", 
    22: "Bumpy Road", 
    23: "Slippery Road", 
    24: "Road Narrows (Right)", 
    25: "Road Work", 
    26: "Traffic Signals", 
    27: "Pedestrian Crossing", 
    28: "Children Crossing", 
    29: "Bicycle Crossing", 
    30: "Beware of Ice/Snow", 
    31: "Wild Animals Crossing", 
    32: "End of Speed & Passing Limits", 
    33: "Turn Right Ahead", 
    34: "Turn Left Ahead", 
    35: "Ahead Only", 
    36: "Go Straight or Right", 
    37: "Go Straight or Left", 
    38: "Keep Right", 
    39: "Keep Left", 
    40: "Roundabout", 
    41: "End of No Overtaking", 
    42: "End of No Overtaking (Trucks)"
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 detection
    results = model(frame)

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            class_id = int(box.cls[0])  # Class ID
            conf = box.conf[0]  # Confidence score

            # Get class name from dictionary
            class_name = class_names.get(class_id, "Unknown Sign")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Traffic Sign Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
