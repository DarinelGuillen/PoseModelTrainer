import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model_path = 'path_to_your_model/keras_model.h5'
model = load_model(model_path)

# Print the model summary
model.summary()

# Save the model architecture to a JSON file
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights('model_weights.h5')
