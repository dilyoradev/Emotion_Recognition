# Emotion_Recognition
This project detects faces from a webcam and recognizes emotions using a deep learning model.
# Emotion Detection with OpenCV & PyTorch

## 📌 Overview
This project is a **real-time emotion detection system** using OpenCV and PyTorch. It captures webcam footage, detects faces, and predicts emotions using a deep learning model.

## 🚀 Features
- **Face Detection**: Uses OpenCV's Haar Cascade for face detection.
- **Emotion Recognition**: Predicts emotions using a trained PyTorch model.
- **Live Webcam Feed**: Displays real-time predictions.
- **Graphics Overlay**: Draws bounding boxes around detected faces and labels them with emotions.

## 🛠 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download the Haar Cascade Model
Make sure you have OpenCV's pre-trained face detection model:
```bash
mkdir models
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml -P models/
```

## 🎥 Usage
### Running the Emotion Detector
```bash
python src/webcam.py
```
Press **'q'** to exit the webcam window.

## 🎯 Emotion Labels
The model predicts the following emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## 📸 Example Screenshot
![Example Screenshot]<img width="200" alt="test_image" src="https://github.com/user-attachments/assets/3578f5dc-6b60-43d7-8651-d61a83eda751" />

## 🔧 Future Improvements
- Improve model accuracy with a larger dataset.
- Add graphical overlays for a more interactive UI.
- Integrate with Flask for a web-based interface.


🔗 **Author:** [Dilyora](https://github.com/dilyoradev)

🤖 **Contributions & Feedback:** PRs and issues are more than welcome!
