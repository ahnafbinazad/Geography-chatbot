import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from class_names import countries  # Importing the list of countries from the class_names file
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

# Define the function to create and compile the CNN model
def create_model(filters=32, kernel_size=(3, 3), dropout_rate=0.5):
    """
    Function to create and compile a Convolutional Neural Network (CNN) model.

    Parameters:
    - filters: Integer, number of filters in the convolutional layers.
    - kernel_size: Tuple, size of the convolutional kernels.
    - dropout_rate: Float, dropout rate for regularization.

    Returns:
    - model: Compiled CNN model.
    """
    inputs = keras.Input(shape=(28, 28, 3))  # Input layer shape
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation="relu")(inputs)  # First convolutional layer
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Max pooling layer
    x = Conv2D(filters=filters * 2, kernel_size=kernel_size, activation="relu")(x)  # Second convolutional layer
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Max pooling layer
    x = Flatten()(x)  # Flatten layer
    x = Dense(256, activation="relu")(x)  # Dense layer with ReLU activation
    x = Dropout(dropout_rate)(x)  # Dropout layer for regularization
    outputs = Dense(num_classes, activation="softmax")(x)  # Output layer with softmax activation
    model = Model(inputs, outputs)  # Creating the model

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Define the hyperparameter search space
param_dist = {
    'filters': [32, 64],                # Number of filters in the convolutional layers
    'kernel_size': [(3, 3), (5, 5)],    # Size of the convolutional kernels
    'dropout_rate': [0.5, 0.6]           # Dropout rate for regularization
}

# Create a custom Keras model class that meets scikit-learn's requirements
class CustomKerasClassifier:
    def __init__(self, filters=32, kernel_size=(3, 3), dropout_rate=0.5):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.model = None

    def create_model(self):
        self.model = create_model(filters=self.filters, kernel_size=self.kernel_size, dropout_rate=self.dropout_rate)

    def fit(self, X, y):
        self.create_model()
        self.model.fit(X, y, verbose=0)
        return self

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

    def get_params(self, deep=True):
        return {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

# Initialize custom Keras model
keras_model = CustomKerasClassifier()

# Perform random search
random_search = RandomizedSearchCV(estimator=keras_model, param_distributions=param_dist, cv=3, verbose=2)
random_search.fit(train_images, train_labels_one_hot)

# Print best parameters found
print("Best Parameters: ", random_search.best_params_)

# Evaluate the best model
best_model = random_search.best_estimator_
test_loss, test_acc = best_model.model.evaluate(test_images, test_labels_one_hot, verbose=2)
plt.imshow(test_images[10])
print('\nTest accuracy:', test_acc)

# Define the path to save the trained model
model_dir = 'model'
model_filename = 'grid_search_training_model.h5'
model_path = os.path.join(model_dir, model_filename)

# Check if the model directory exists, if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model with the new filename
best_model.model.save(model_path)
print("Model saved successfully.")
