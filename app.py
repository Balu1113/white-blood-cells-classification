# Install necessary libraries
!pip install kaggle
!pip install pillow
!pip install scikit-learn

import os
import zipfile
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Mount Google Drive to save files
from google.colab import drive
drive.mount('/content/drive')
# Upload the Kaggle API token
from google.colab import files
files.upload()  # This will prompt you to upload the kaggle.json file

# Move the kaggle.json file to the proper location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the BCCD dataset from Kaggle
!kaggle datasets download -d paultimothymooney/blood-cells

# Unzip the dataset and suppress prompts
!unzip -qo blood-cells.zip -d blood-cells
# Define the path to the dataset
dataset_path = 'blood-cells/dataset2-master/dataset2-master/images/TRAIN'

# Initialize lists to store images and labels
images = []
labels = []

# Define a function to load images from a folder
def load_images_from_folder(folder, label, sample_size=2000):
    count = 0
    for filename in os.listdir(folder):
        if count >= sample_size:
            break
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((96, 96))  # Resize image to a smaller size to save memory
        images.append(np.array(img))
        labels.append(label)
        count += 1

# Load images from each folder (Eosinophil, Lymphocyte, Monocyte, Neutrophil)
load_images_from_folder(os.path.join(dataset_path, 'EOSINOPHIL'), 0, sample_size=2000)
load_images_from_folder(os.path.join(dataset_path, 'LYMPHOCYTE'), 1, sample_size=2000)
load_images_from_folder(os.path.join(dataset_path, 'MONOCYTE'), 2, sample_size=2000)
load_images_from_folder(os.path.join(dataset_path, 'NEUTROPHIL'), 3, sample_size=2000)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

print(f'Loaded {len(images)} images.')

# Normalize the images
images = images / 255.0

# Save the images and labels as pickle files
with open('/content/wbc_images.pkl', 'wb') as f:
    pickle.dump(images, f)

with open('/content/wbc_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print('Data saved.')
# Load the preprocessed data
with open('/content/wbc_images.pkl', 'rb') as f:
    images = pickle.load(f)

with open('/content/wbc_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# Convert labels to categorical
from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, num_classes=4)

# Split the data into training, validation, and test sets
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define a deeper CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/content/best_cnn_model.h5', save_best_only=True, monitor='val_loss')

# Train the CNN model with data augmentation
cnn_history = cnn_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])
# Load the best model
cnn_model.load_weights('/content/best_cnn_model.h5')

# Evaluate the CNN model on the test set
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f'CNN Test accuracy: {cnn_test_accuracy}')

# Generate classification report for CNN
from sklearn.metrics import classification_report

y_pred_cnn = cnn_model.predict(X_test)
y_pred_classes_cnn = np.argmax(y_pred_cnn, axis=1)
y_true_cnn = np.argmax(y_test, axis=1)

print('Classification Report - CNN')
print(classification_report(y_true_cnn, y_pred_classes_cnn))

# Plot CNN training history
plt.plot(cnn_history.history['accuracy'], label='accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('CNN Training History')
plt.show()

plt.plot(cnn_history.history['loss'], label='loss')
plt.plot(cnn_history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.title('CNN Loss History')
plt.show()
# Flatten images for non-CNN algorithms
flat_images = images.reshape((images.shape[0], -1))

# Split the data into training, validation, and test sets for non-CNN algorithms
X_train_flat, X_temp_flat, y_train_flat, y_temp_flat = train_test_split(flat_images, labels, test_size=0.3, random_state=42)
X_val_flat, X_test_flat, y_val_flat, y_test_flat = train_test_split(X_temp_flat, y_temp_flat, test_size=0.5, random_state=42)

# Convert labels back to single integer format for scikit-learn
y_train_flat = np.argmax(y_train_flat, axis=1)
y_val_flat = np.argmax(y_val_flat, axis=1)
y_test_flat = np.argmax(y_test_flat, axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_flat, y_train_flat)

y_pred_rf = rf_model.predict(X_test_flat)

print('Classification Report - Random Forest')
print(classification_report(y_test_flat, y_pred_rf))
print(f'Accuracy: {accuracy_score(y_test_flat, y_pred_rf)}')
from sklearn.svm import SVC

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_flat, y_train_flat)

y_pred_svm = svm_model.predict(X_test_flat)

print('Classification Report - SVM')
print(classification_report(y_test_flat, y_pred_svm))
print(f'Accuracy: {accuracy_score(y_test_flat, y_pred_svm)}')

