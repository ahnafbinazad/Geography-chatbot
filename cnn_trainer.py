import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from class_names import countries  # Importing the list of countries from the class_names file

# Load preprocessed train and test images
train_images = np.load('./data/train_images.npy')
test_images = np.load('./data/test_images.npy')
train_labels = np.load('./data/train_labels.npy')
test_labels = np.load('./data/test_labels.npy')

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert encoded labels to one-hot encoding
num_classes = len(label_encoder.classes_)
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=num_classes)
test_labels_one_hot = to_categorical(test_labels_encoded, num_classes=num_classes)

# Define the CNN model
model = keras.Sequential(
    [
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax")
    ]
)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels_one_hot, epochs=10, batch_size=128)

# Evaluate the trained model
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot, verbose=2)
plt.imshow(test_images[10])
print('\nTest accuracy:', test_acc)

# Define the path to save the trained model
model_dir = 'model'
model_filename = 'trained_model.h5'
model_path = os.path.join(model_dir, model_filename)

# Check if the model directory exists, if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model.save(model_path)
print("Model saved successfully.")
