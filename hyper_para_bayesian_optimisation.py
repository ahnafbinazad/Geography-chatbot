import numpy as np
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

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
def create_model(filters=32, kernel_size=(3, 3), dropout_rate=0.5):
    """
    Create a convolutional neural network (CNN) model.

    Args:
        filters (int): Number of filters in convolutional layers.
        kernel_size (tuple): Size of convolutional kernels.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        keras.Model: Compiled Keras model.
    """
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
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels_one_hot, test_size=0.2, random_state=42)

# Define hyperparameters for grid search
param_grid = {
    'filters': [16, 32, 64],
    'kernel_size': [(3, 3), (5, 5)],
    'dropout_rate': [0.2, 0.5, 0.7]
}

# Create a custom Keras classifier
class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, filters=32, kernel_size=(3, 3), dropout_rate=0.5):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.model = None

    def fit(self, X, y):
        self.model = create_model(self.filters, self.kernel_size, self.dropout_rate)
        self.model.fit(X, y, epochs=10, batch_size=128)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

    def score(self, X, y):
        _, accuracy = self.model.evaluate(X, y)
        return accuracy

# Create the grid search object
grid_search = GridSearchCV(estimator=CustomKerasClassifier(), param_grid=param_grid, cv=3, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_acc = best_model.score(test_images, test_labels_one_hot)
print('\nTest accuracy:', test_acc)
