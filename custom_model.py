import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Load the model architecture
with open('model_architecture.json', 'r') as json_file:
    model_json = json_file.read()

# Define the model
model = model_from_json(model_json)

# Load the weights
model.load_weights('model_weights.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to verify
model.summary()

# Now you can use the model as usual
# For example, to make predictions:
# predictions = model.predict(some_data)
