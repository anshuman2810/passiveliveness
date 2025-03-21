from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import cv2

app = Flask(__name__)

# Load the trained model
class LivenessNet(torch.nn.Module):
    def __init__(self):
        super(LivenessNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, 2)  # 2 classes: live and spoof

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = LivenessNet()
model.load_state_dict(torch.load("best_liveness_model.pth"))
model.eval()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing function
def preprocess_face(face):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    face = Image.fromarray(face)
    face = transform(face).unsqueeze(0)
    return face

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle liveness detection
@app.route("/detect", methods=["POST"])
def detect():
    # Get the image from the request
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])  # Decode base64 image

    # Convert image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    result_label = "No Face Detected"
    result_color = (0, 0, 255)  # Default to red (spoof)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Preprocess the face
        face_tensor = preprocess_face(face)

        # Perform liveness detection
        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            is_live = predicted.item() == 1  # Assuming 1 is the label for "live"

        # Set the result label and color
        result_label = "Live" if is_live else "Spoof"
        result_color = (0, 255, 0) if is_live else (0, 0, 255)  # Green for live, red for spoof

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), result_color, 2)
        cv2.putText(frame, result_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_color, 2)

    # Convert the processed frame to base64
    _, buffer = cv2.imencode(".jpg", frame)
    processed_image = base64.b64encode(buffer).decode("utf-8")

    # Return the processed image and result
    return jsonify({
        "image": processed_image,
        "result": result_label
    })

if __name__ == "__main__":
    app.run(debug=True)