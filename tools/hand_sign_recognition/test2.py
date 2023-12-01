import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Load model and processor
processor = AutoImageProcessor.from_pretrained("model2")
model = AutoModelForImageClassification.from_pretrained("model2")
model.eval()  # Set the model to evaluation mode

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the label map based on the provided config
id2label = {
    "0": "A", "1": "B", "10": "K", "11": "L", "12": "M", "13": "N", "14": "O",
    "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T", "2": "C", "20": "U",
    "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z", "26": "del",
    "27": "nothing", "28": "space", "3": "D", "4": "E", "5": "F", "6": "G",
    "7": "H", "8": "I", "9": "J"
}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR (OpenCV format) to RGB (model format)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image to be compatible with the model
    inputs = processor(images=rgb_image, return_tensors="pt")

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label
    predicted_label_id = outputs.logits.argmax(-1).item()
    predicted_label = id2label[str(predicted_label_id)]

    # Draw a bounding box around the entire frame
    height, width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (width, height), (255, 0, 0), 2)

    # Put the predicted label on the frame
    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Webcam - Press Q to Quit', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
cap.release()
cv2.destroyAllWindows()
