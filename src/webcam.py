import cv2
import torch
import numpy as np
from model import create_model
from save_load import load_model
from config import DEVICE
import torchvision.transforms as transforms
from PIL import Image

# Load trained model
model = create_model()
load_model(model)
model.to(DEVICE)
model.eval()  # Set model to evaluation mode

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")

# Emotion labels (modify based on your dataset)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        
        # Convert face to RGB (OpenCV loads in BGR)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and apply transformations
        img = Image.fromarray(face_rgb)
        img = transform(img).unsqueeze(0).to(DEVICE)  # Apply transforms and add batch dimension

        # Get prediction
        with torch.no_grad():
            output = model(img)
            predicted_class = torch.argmax(output, dim=1).item()

        # Display prediction on screen
        emotion = emotion_labels[predicted_class]

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put emotion label near face
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
