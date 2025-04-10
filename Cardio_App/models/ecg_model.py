import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(42)

# Define dataset directory
dataset_dir = 'data'
normal_dir = os.path.join(dataset_dir, 'normal')
abnormal_dir = os.path.join(dataset_dir, 'abnormal')
infarction_dir = os.path.join(dataset_dir, 'myocardial')  # Fixed folder name

image_paths = []
labels = []

# Load image paths and labels
for category, directory in zip(['normal', 'abnormal', 'myocardial'],
                               [normal_dir, abnormal_dir, infarction_dir]):
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found!")
        continue
    for image_filename in os.listdir(directory):
        image_paths.append(os.path.join(directory, image_filename))
        labels.append(category)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.20, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Image size
target_size = (224, 224)


def preprocess_image(image_path, target_size):
    """Read and preprocess an image."""
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found!")
        return np.zeros((*target_size, 3))  # Return blank image if missing
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read {image_path}")
        return np.zeros((*target_size, 3))  # Return blank image if read fails
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values
    return image


# Preprocess images
X_train_preprocessed = np.array([preprocess_image(img, target_size) for img in X_train])
X_val_preprocessed = np.array([preprocess_image(img, target_size) for img in X_val])
X_test_preprocessed = np.array([preprocess_image(img, target_size) for img in X_test])

# Convert labels to numeric values
label_encoder = LabelEncoder()
y_train_array = label_encoder.fit_transform(y_train)
y_val_array = label_encoder.transform(y_val)
y_test_array = label_encoder.transform(y_test)

# Save label encoder
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))

# Define CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

# Compile model
cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train model
cnn_model.fit(X_train_preprocessed, y_train_array,
              validation_data=(X_val_preprocessed, y_val_array),
              epochs=16, batch_size=32)

# Save trained model
cnn_model.save(os.path.join(models_dir, 'ecg_model.keras'))

print("Model and label encoder saved successfully!")
