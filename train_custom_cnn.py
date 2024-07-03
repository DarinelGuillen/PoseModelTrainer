import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import matplotlib.pyplot as plt

# Directorios
base_dir = 'C:/Users/darin/Documents/8B/tensorflow/data'
train_dir = base_dir
charts_dir = 'C:/Users/darin/Documents/8B/tensorflow/charts'

# Parámetros de imagen
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation y normalización para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

# Generador de datos para entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Generador de datos para validación
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Imprimir índices de clases
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# Definir un modelo CNN personalizado con dropout para regularización
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
        self.dropout = layers.Dropout(0.5)
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

# Obtener número de clases
num_classes = len(train_generator.class_indices)

# Instanciar el modelo
model = CustomCNN(num_classes=num_classes)

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Función para registrar detalles del entrenamiento
def log_training_details(log_file, history, class_indices, val_accuracy):
    with open(log_file, 'a') as f:
        f.write(f"Training log for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Class indices: {class_indices}\n")
        f.write(f"Epochs: {len(history.history['accuracy'])}\n")
        f.write(f"Accuracy per epoch: {history.history['accuracy']}\n")
        f.write(f"Validation accuracy per epoch: {history.history['val_accuracy']}\n")
        f.write(f"Final validation accuracy: {val_accuracy * 100:.2f}%\n")
        f.write(f"{'='*50}\n\n")

# Función para guardar una gráfica de la precisión de entrenamiento y validación
def save_accuracy_plot(history, save_dir):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{save_dir}/accuracy_plot.png")

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=3,  # Reduce the number of epochs for quick testing
    validation_data=validation_generator
)

# Guardar el modelo entrenado en formato SavedModel
model_save_path = 'custom_cnn_model_saved'
model.save(model_save_path, save_format='tf')
print(f"Model saved to {model_save_path}")

# Evaluar el modelo
loss, acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {acc * 100:.2f}%")

# Registrar detalles del entrenamiento
log_file = 'training_log.md'
log_training_details(log_file, history, class_indices, acc)

# Guardar la gráfica de precisión
os.makedirs(charts_dir, exist_ok=True)
save_accuracy_plot(history, charts_dir)
print(f"Accuracy plot saved to {charts_dir}")
