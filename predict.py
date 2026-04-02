import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF logging

import tf_keras as keras # Use tf_keras (Keras 2) for better compatibility
from tf_keras.models import load_model
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
print("Loading model using tf_keras (Keras 2 compatibility)...")
try:
    # tf_keras should handle the Teachable Machine h5 file correctly
    model = load_model("keras_model.h5", compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load the labels
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("labels.txt not found. Using default labels.")
    class_names = ["0 Class 1", "1 Class 2"]

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("Starting webcam... Press 'Esc' to exit.")

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    if not ret:
        print("Failed to grab frame from camera. Exiting...")
        break

    # --- PREPROCESSING FIX ---
    # 1. Convert BGR to RGB (Models usually expect RGB, OpenCV decodes to BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Center crop to square (Maintains aspect ratio, prevents squashing)
    h, w = image_rgb.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    image_cropped = image_rgb[start_h:start_h + min_dim, start_w:start_w + min_dim]
    
    # 3. Resize the raw image into (224-height,224-width) pixels
    image_resized = cv2.resize(image_cropped, (224, 224), interpolation=cv2.INTER_AREA)

    # 4. Make the image a numpy array and reshape it to the models input shape.
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # 5. Normalize the image array to [-1, 1]
    image_array = (image_array / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Clean up class name for display
    # Assumes label format "0 Name" or similar
    name_only = class_name[2:] if len(class_name) > 2 else class_name
    display_text = f"{name_only}: {np.round(confidence_score * 100, 1)}%"
    
    # Print prediction and confidence score to console
    print(f"\rPrediction: {display_text}", end="")

    # Add text to the image
    cv2.putText(image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show the image in a window
    cv2.imshow("Detection (Press ESC to exit)", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

print("\nExiting...")
camera.release()
cv2.destroyAllWindows()


