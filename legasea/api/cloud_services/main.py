from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort
import os
import random


app = FastAPI()

# Define model paths
MODEL_DIR = "models/aerial_detection"
MODEL_PATH = os.path.join(MODEL_DIR, "waldo_v3.onnx")

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

# WALDO classes and configuration
NAMES = ['LightVehicle', 'Person', 'Building', 'UPole', 'Boat', 'Bike', 'Container', 'Truck', 'Gastank', 'Digger', 'Solarpanels', 'Bus']
COLORS = {name:[random.randint(0, 255) for _ in range(3)] for name in NAMES}

def resize_and_pad(frame, expected_width, expected_height):
    ratio = min(expected_width / frame.shape[1], expected_height / frame.shape[0])
    new_width = int(frame.shape[1] * ratio)
    new_height = int(frame.shape[0] * ratio)
    frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.zeros((expected_height, expected_width, 3), dtype=np.uint8)
    y_offset = (expected_height - new_height) // 2
    x_offset = (expected_width - new_width) // 2
    padded_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
    return padded_frame

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = resize_and_pad(img, 960, 960)
    return img

def process_frame(frame, session, confidence_threshold=0.5):
    # Prepare image for inference
    image = frame.copy()
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255

    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: im})[0]

    # Process detections
    detections = []
    for output in outputs:
        confidence = float(output[6])
        if confidence > confidence_threshold:
            class_id = int(output[5])
            bbox = output[1:5].astype(np.int32)
            detections.append({
                "class": NAMES[class_id],
                "confidence": confidence,
                "bbox": bbox.tolist()
            })

    return detections


def process_detections(frame, detections):
    annotated_frame = frame.copy()
    for detection in detections:
        bbox = detection["bbox"]
        label = detection["class"]
        conf = detection["confidence"]
        color = COLORS[label]
        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_frame

@app.get("/")
async def root():
    return {"message": "WALDO API is running"}

@app.post("/detect/")
async def detect_object(file: UploadFile = File(...)):
    image = Image.open(file.file)
    processed_image = preprocess_image(image)

    # Initialize model session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(os.path.join(MODEL_DIR, "waldo_v3.onnx"), providers=providers)

    # Process image and get detections
    detections = process_frame(processed_image, session)

    # Annotate image with detections
    annotated_image = process_detections(processed_image, detections)

    return {
        "filename": file.filename,
        "detections": detections,
        "processed_shape": processed_image.shape,
        "detected_objects_count": len(detections)
    }
