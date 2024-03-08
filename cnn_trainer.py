import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from class_names import countries  # Importing the list of countries from the class_names file

# Load preprocessed train and test images
train_images = np.load('./data/train_images.npy')  # Loading preprocessed training images
test_images = np.load('./data/test_images.npy')    # Loading preprocessed testing images
train_labels = np.load('./data/train_labels.npy')  # Loading training labels
test_labels = np.load('./data/test_labels.npy')    # Loading testing labels

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()  # Initializing LabelEncoder object
train_labels_encoded = label_encoder.fit_transform(train_labels)    # Encoding training labels
test_labels_encoded = label_encoder.transform(test_labels)          # Encoding testing labels

# Convert encoded labels to one-hot encoding
num_classes = len(label_encoder.classes_)  # Getting the number of unique classes
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=num_classes)  # One-hot encoding training labels
test_labels_one_hot = to_categorical(test_labels_encoded, num_classes=num_classes)    # One-hot encoding testing labels

# Define the CNN model
model = keras.Sequential(
    [
        # Convolutional layer with ReLU activation
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 3)),
        # Max pooling layer
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Convolutional layer with ReLU activation
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        # Max pooling layer
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten layer
        keras.layers.Flatten(),
        # Dense layer with ReLU activation
        keras.layers.Dense(256, activation="relu"),
        # Dropout layer for regularization
        keras.layers.Dropout(0.5),
        # Output layer with softmax activation
        keras.layers.Dense(num_classes, activation="softmax")
    ]
)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels_one_hot, epochs=10, batch_size=128)  # Fitting the model to the training data

# Evaluate the trained model
test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot, verbose=2)  # Evaluating the model on test data
plt.imshow(test_images[10])  # Displaying an example image from the test set
print('\nTest accuracy:', test_acc)  # Printing the test accuracy

# Define the path to save the trained model
model_dir = 'model'  # Directory to save the model
model_filename = 'trained_model.h5'  # Filename for the saved model
model_path = os.path.join(model_dir, model_filename)  # Creating the full path to save the model

# Check if the model directory exists, if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)  # Creating the directory if it doesn't exist

# Save the trained model
model.save(model_path)  # Saving the trained model to the specified path
print("Model saved successfully.")  # Printing confirmation message
