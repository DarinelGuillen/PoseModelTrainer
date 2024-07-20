import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import mediapipe as mp

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

# Registrar la clase del modelo
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

# Cargar el modelo y los pesos guardados
model_path = 'C:/Users/darin/Documents/8B/tensorflow/modelo_entrenado'
model_json_path = f'{model_path}/model.json'
model_weights_path = f'{model_path}/weights.weights.h5'

with open(model_json_path, 'r') as json_file:
    model_json = json_file.read()

pose_classification_model = tf.keras.models.model_from_json(model_json)
pose_classification_model.load_weights(model_weights_path)

# Etiquetas de las clases
class_labels = [
    "1 _Up_Hand",
    "10 _Down_View",
    "11 _X_Pose",
    "12 _Forward_Step",
    "13 _Attention_Forward",
    "2 _V_Pose",
    "3 _Signal_Right",
    "4 _Double_Bicep",
    "5 _Side_Arm",
    "6 _Stop_Hands",
    "7 _Neutral",
    "8 _T_Pose",
    "9 _L_Pose"
]

# Función para preprocesar la imagen
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = np.array(image)
    return image

# Función para detectar pose y extraer puntos clave
def detect_pose(image):
    results = pose.process(image)
    if not results.pose_landmarks:
        return np.zeros((33, 2)).flatten()

    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((landmark.x, landmark.y))
    keypoints = np.array(keypoints).flatten()
    return keypoints

# Captura de video desde la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = preprocess_image(frame)
    keypoints = detect_pose(image)

    # Hacer una predicción
    keypoints_input = np.expand_dims(keypoints, axis=0)
    prediction = pose_classification_model.predict(keypoints_input)
    predicted_class = class_labels[np.argmax(prediction)]

    # Mostrar la predicción en la imagen
    cv2.putText(frame, f'Prediccion: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Mostrar la imagen con la predicción
    cv2.imshow('Pose Classification', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
