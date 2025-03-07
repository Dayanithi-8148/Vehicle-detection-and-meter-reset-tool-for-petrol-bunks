import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np

cap = cv2.VideoCapture("./example.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

left_boundary = 500
right_boundary = 1100
top_boundary = 200
bottom_boundary = 600

def detect_objects(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    return results
vehicles=["car","truck","motorcycle","bike","bus"]

def draw_boxes(frame, results):
    inside = False
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] not in vehicles:
            continue
        box = [round(i, 2) for i in box.tolist()]
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (0, 255, 0)  
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        cv2.putText(frame, text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        if start_point[0] > left_boundary and end_point[0] < right_boundary and start_point[1] > top_boundary and end_point[1] < bottom_boundary:
            inside = True
    
    status = "Inside" if inside else "Out"
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    return frame

def draw_3d_boundary(frame):
    # Draw vertical lines
    cv2.line(frame, (left_boundary, 0), (left_boundary, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (right_boundary, 0), (right_boundary, frame.shape[0]), (255, 0, 0), 2)
    
    # Draw horizontal lines
    cv2.line(frame, (0, top_boundary), (frame.shape[1], top_boundary), (255, 0, 0), 2)
    cv2.line(frame, (0, bottom_boundary), (frame.shape[1], bottom_boundary), (255, 0, 0), 2)
    
    # Draw 3D effect
    cv2.line(frame, (left_boundary, top_boundary), (left_boundary, bottom_boundary), (0, 0, 255), 2)
    cv2.line(frame, (right_boundary, top_boundary), (right_boundary, bottom_boundary), (0, 0, 255), 2)
    cv2.line(frame, (left_boundary, top_boundary), (right_boundary, top_boundary), (0, 0, 255), 2)
    cv2.line(frame, (left_boundary, bottom_boundary), (right_boundary, bottom_boundary), (0, 0, 255), 2)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read frame")
        break
    
    if frame is None or frame.size == 0:
        print("Empty frame")
        continue
    
    print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")

    results = detect_objects(frame)
    
    frame = draw_boxes(frame, results)
    frame = draw_3d_boundary(frame)
    
    cv2.imshow('Real-Time Vehicle Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
