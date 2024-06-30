import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Suppress TensorFlow logging if necessary
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the model
model_path = 'C:/Users/darin/Documents/8B/C3_RN_TESTING/tensorflow/keras_model.h5'
labels_path = 'C:/Users/darin/Documents/8B/C3_RN_TESTING/tensorflow/labels.txt'
model = load_model(model_path, compile=False)

# Load the labels
with open(labels_path, "r") as f:
    class_names = f.readlines()

# Initialize the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image")
            break

        # Convert the captured frame into PIL format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image to 224x224
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)

        # Convert the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Run the inference
        prediction = model.predict(data)
        index = np.argmax(prediction)
        predicted_label = class_names[index].strip()

        # Display the resulting frame
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
