import numpy as np
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

# Load preprocessed train and test images
# Loading the preprocessed images and labels from the provided files
train_images = np.load('./data/train_images.npy')
test_images = np.load('./data/test_images.npy')
train_labels = np.load('./data/train_labels.npy')
test_labels = np.load('./data/test_labels.npy')

# Encode labels using LabelEncoder
# Using LabelEncoder to encode the categorical labels into numerical values
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert encoded labels to one-hot encoding
# Convert the numerical encoded labels to one-hot encoding
num_classes = len(label_encoder.classes_)
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=num_classes)
test_labels_one_hot = to_categorical(test_labels_encoded, num_classes=num_classes)

# Define the CNN model
# Defining a convolutional neural network (CNN) model using Keras Sequential API
def create_model(filters=32, kernel_size=(3, 3), dropout_rate=0.5):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", input_shape=(28, 28, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size, activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Split train data into train and validation sets
# Splitting the training data into training and validation sets for model evaluation
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels_one_hot, test_size=0.2, random_state=42)

# Define hyperparameters for grid search
# Defining the hyperparameters to search over using grid search
param_grid = {
    'filters': [16, 32, 64],                # Number of filters in convolutional layers
    'kernel_size': [(3, 3), (5, 5)],        # Size of convolutional kernels
    'dropout_rate': [0.2, 0.5, 0.7]          # Dropout rate for regularization
}

# Create a custom Keras classifier
# Creating a custom Keras classifier compatible with scikit-learn
class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, filters=32, kernel_size=(3, 3), dropout_rate=0.5):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.model = None

    def fit(self, X, y):
        # Creating the model with specified hyperparameters and fitting it to the data
        self.model = create_model(self.filters, self.kernel_size, self.dropout_rate)
        self.model.fit(X, y, epochs=10, batch_size=128)  # Training for 10 epochs with batch size 128
        return self

    def predict(self, X):
        # Making predictions using the trained model
        return np.argmax(self.model.predict(X), axis=-1)

    def score(self, X, y):
        # Evaluating the model's accuracy on the given data
        _, accuracy = self.model.evaluate(X, y)
        return accuracy

# Create the grid search object
# Creating a grid search object to find the best hyperparameters
grid_search = GridSearchCV(estimator=CustomKerasClassifier(), param_grid=param_grid, cv=3, verbose=2)

# Fit the grid search to the data
# Fitting the grid search to the training data to find the best model
grid_search.fit(X_train, y_train)

# Get the best model
# Retrieving the best model found by the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
# Evaluating the best model's performance on the test set
test_acc = best_model.score(test_images, test_labels_one_hot)
print('\nTest accuracy:', test_acc)
