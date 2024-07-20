import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Directory paths
base_dir = 'C:/Users/darin/Documents/8B/tensorflow/dataTEST'
train_dir = base_dir  # Assuming all images are in the 'data' directory

# Image parameters
img_height, img_width = 224, 224  # Adjust as necessary
batch_size = 32

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # Added rotation for augmentation
    width_shift_range=0.2,  # Added width shift for augmentation
    height_shift_range=0.2,  # Added height shift for augmentation
    validation_split=0.2  # Use 20% of the data for validation
)

# Data generator for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

# Data generator for validation
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Print class indices
print("Class indices:", train_generator.class_indices)

# Define a custom CNN model with dropout for regularization
class CustomCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)  # Added dropout for regularization
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

# Get number of classes from the generator
num_classes = len(train_generator.class_indices)

# Instantiate the model
model = CustomCNN(num_classes=num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Directory to save the model files
save_dir = 'C:/Users/darin/Documents/8B/tensorflow/modelo_entrenado'
os.makedirs(save_dir, exist_ok=True)

# Save the model architecture
model_json_path = os.path.join(save_dir, 'model.json')
model_json = model.to_json()
with open(model_json_path, 'w') as json_file:
    json_file.write(model_json)

# Save the model weights
model_weights_path = os.path.join(save_dir, 'weights.weights.h5')
model.save_weights(model_weights_path)

# Save metadata
metadata = {
    'model_name': 'custom_cnn_model',
    'input_shape': (img_height, img_width, 3),
    'num_classes': num_classes,
    'training_samples': train_generator.samples,
    'validation_samples': validation_generator.samples,
    'epochs': 10
}

metadata_path = os.path.join(save_dir, 'metadata.json')
with open(metadata_path, 'w') as metadata_file:
    json.dump(metadata, metadata_file)

print(f"Model architecture saved to {model_json_path}")
print(f"Model weights saved to {model_weights_path}")
print(f"Metadata saved to {metadata_path}")

# Evaluate the model
loss, acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {acc * 100:.2f}%")
