import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import matplotlib.pyplot as plt

# Directory paths
base_dir = 'C:/Users/darin/Documents/8B/tensorflow/data'
train_dir = base_dir  # Assuming all images are in the 'data' directory
charts_dir = 'C:/Users/darin/Documents/8B/tensorflow/charts'

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
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

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

# Function to log training details
def log_training_details(log_file, history, class_indices, val_accuracy):
    with open(log_file, 'a') as f:
        f.write(f"Training log for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Class indices: {class_indices}\n")
        f.write(f"Epochs: {len(history.history['accuracy'])}\n")
        f.write(f"Accuracy per epoch: {history.history['accuracy']}\n")
        f.write(f"Validation accuracy per epoch: {history.history['val_accuracy']}\n")
        f.write(f"Final validation accuracy: {val_accuracy * 100:.2f}%\n")
        f.write(f"{'='*50}\n\n")

# Function to save a plot of the training and validation accuracy
def save_accuracy_plot(history, save_dir):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{save_dir}/accuracy_plot.png")

# Train the model
history = model.fit(
    train_generator,
    epochs=2,  # Reduced epochs for quicker testing
    validation_data=validation_generator
)

# Save the trained model
model_save_path = 'custom_cnn_model.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate the model
loss, acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {acc * 100:.2f}%")

# Log the training details
log_file = 'training_log.md'
log_training_details(log_file, history, class_indices, acc)

# Save the accuracy plot
os.makedirs(charts_dir, exist_ok=True)
save_accuracy_plot(history, charts_dir)
print(f"Accuracy plot saved to {charts_dir}")
