import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import mediapipe as mp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Directorio para guardar los archivos del modelo
save_dir = 'C:/Users/darin/Documents/8B/tensorflow/modelo_entrenado'
os.makedirs(save_dir, exist_ok=True)

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

# Preprocesamiento de datos
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0  # Normalizar la imagen
    return image

def detect_pose(image):
    image_rgb = tf.image.convert_image_dtype(image, tf.uint8)
    image_rgb = np.array(image_rgb)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return np.zeros((33, 2)).flatten()

    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((landmark.x, landmark.y))
    keypoints = np.array(keypoints).flatten()
    return keypoints

# Directorios de imágenes
base_dir = 'C:/Users/darin/Documents/8B/tensorflow/dataTEST'

# Crear generadores de datos con división de validación
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Definir la red neuronal
@tf.keras.utils.register_keras_serializable()
class PoseClassificationModel(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(PoseClassificationModel, self).__init__(**kwargs)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        return self.dense2(x)

    def get_config(self):
        config = super(PoseClassificationModel, self).get_config()
        config.update({"num_classes": self.dense2.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


num_classes = len(train_generator.class_indices)
pose_classification_model = PoseClassificationModel(num_classes=num_classes)

pose_classification_model.compile(optimizer='adam',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

# Función para extraer los puntos clave de una imagen
def extract_keypoints(image_path):
    image = preprocess_image(image_path)
    keypoints = detect_pose(image)
    return keypoints

# Crear datasets de entrenamiento y validación
train_images = train_generator.filepaths
val_images = validation_generator.filepaths

train_keypoints = np.array([extract_keypoints(img_path) for img_path in train_images])
val_keypoints = np.array([extract_keypoints(img_path) for img_path in val_images])

train_labels = tf.keras.utils.to_categorical(train_generator.classes, num_classes=num_classes)
val_labels = tf.keras.utils.to_categorical(validation_generator.classes, num_classes=num_classes)

# Entrenar el modelo
history = pose_classification_model.fit(
    train_keypoints,
    train_labels,
    epochs=10,
    validation_data=(val_keypoints, val_labels)
)

# Guardar el modelo, pesos y metadatos
model_json_path = os.path.join(save_dir, 'model.json')
model_json = pose_classification_model.to_json()
with open(model_json_path, 'w') as json_file:
    json_file.write(model_json)

model_weights_path = os.path.join(save_dir, 'weights.weights.h5')
pose_classification_model.save_weights(model_weights_path)

metadata = {
    'model_name': 'pose_classification_model',
    'input_shape': (224, 224, 3),
    'num_classes': num_classes,
    'training_samples': len(train_images),
    'validation_samples': len(val_images),
    'epochs': 10
}

metadata_path = os.path.join(save_dir, 'metadata.json')
with open(metadata_path, 'w') as metadata_file:
    json.dump(metadata, metadata_file)

print(f"Model architecture saved to {model_json_path}")
print(f"Model weights saved to {model_weights_path}")
print(f"Metadata saved to {metadata_path}")

# Evaluar el modelo
loss, acc = pose_classification_model.evaluate(val_keypoints, val_labels)
print(f"Validation accuracy: {acc * 100:.2f}%")

# Matriz de confusión
y_pred = np.argmax(pose_classification_model.predict(val_keypoints), axis=1)
cm = confusion_matrix(np.argmax(val_labels, axis=1), y_pred)

# Guardar la matriz de confusión
charts_dir = os.path.join(save_dir, 'charts')
os.makedirs(charts_dir, exist_ok=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_generator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)

today = datetime.today().strftime('%Y-%m-%d')
conf_matrix_path = os.path.join(charts_dir, f'matriz_de_confusion_{today}.png')
plt.savefig(conf_matrix_path)
plt.close()

print(f"Confusion matrix saved to {conf_matrix_path}")
