import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Directorios
base_dir = 'C:/Users/darin/Documents/8B/tensorflow/data'
train_dir = base_dir

# Parámetros de imagen
img_height, img_width = 224, 224
batch_size = 16
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

# Función de preprocesamiento de imágenes
def preprocess_image(image):
    image = np.array(image)
    resized = cv2.resize(image, (img_height, img_width))
    return Image.fromarray(resized)

# Transformaciones para entrenamiento y validación
transform = transforms.Compose([
    transforms.Lambda(preprocess_image),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
dataset = datasets.ImageFolder(train_dir, transform=transform)

# Crear índices de los datos y dividir en entrenamiento y validación
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Verificar que no haya solapamiento entre los índices de entrenamiento y validación
print("Solapamiento en índices de entrenamiento y validación:", set(train_indices).intersection(val_indices))

# Crear samplers para dividir los datos
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Verificar la distribución de las clases en los conjuntos de entrenamiento y validación
train_labels = [dataset[i][1] for i in train_indices]
val_labels = [dataset[i][1] for i in val_indices]
from collections import Counter
print("Distribución de clases en el conjunto de entrenamiento:", Counter(train_labels))
print("Distribución de clases en el conjunto de validación:", Counter(val_labels))

# Definición del modelo en PyTorch
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 26 * 26, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

num_classes = len(dataset.classes)
model = CustomCNN(num_classes=num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 3  # Aumentar el número de épocas para observar el comportamiento
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Evaluación y generación de la matriz de confusión
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Verificar la distribución de las clases en el conjunto de validación
print("Distribución de clases en el conjunto de validación:", Counter(all_labels))

# Inspeccionar algunas predicciones y etiquetas reales
print("Ejemplos de predicciones y etiquetas reales:")
print("Predicciones:", all_preds[:10])
print("Etiquetas Reales:", all_labels[:10])

# Crear la matriz de confusión
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)

# Guardar la gráfica de confusión en una carpeta
charts_dir = 'charts'
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)

# Encontrar un nombre de archivo que no sobrescriba los existentes
file_index = 1
file_path = os.path.join(charts_dir, f'confusion_matrix_{file_index}.png')
while os.path.exists(file_path):
    file_index += 1
    file_path = os.path.join(charts_dir, f'confusion_matrix_{file_index}.png')

plt.savefig(file_path)
plt.show()

# Verificar el tamaño del dataset y las particiones
print(f"Tamaño total del dataset: {dataset_size}")
print(f"Tamaño del conjunto de entrenamiento: {len(train_indices)}")
print(f"Tamaño del conjunto de validación: {len(val_indices)}")

# Mostrar algunos ejemplos de imágenes y etiquetas
examples = enumerate(val_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(f"Ejemplo de etiquetas: {example_targets[:10]}")
