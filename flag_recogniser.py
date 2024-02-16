import os
import numpy as np
from PIL import Image
from tensorflow import keras
import sys
import contextlib

from text_to_speech import text_to_speech


# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to load the trained model
def load_model():
    model_path = 'model/trained_model.h5'
    model = keras.models.load_model(model_path)
    return model


# Function to classify the image
def classify_image(image_path, model, class_names):
    img_array = preprocess_image(image_path)

    # Suppress print output during model prediction
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        prediction = model.predict(img_array)

    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name


def flag_recogniser(voiceEnabled):
    # Ask the user for a file path
    file_path = input("Please enter the file path of the image: ")

    # Check if the file exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    # Load the trained model
    model = load_model()

    # Load the class names
    from class_names import countries

    # Classify the image
    predicted_class_name = classify_image(file_path, model, countries)

    # Print the predicted class
    output = f"That is the flag of {predicted_class_name}"

    print(output)
    text_to_speech(voiceEnabled, output)


# if __name__ == "__main__":
#     while True:
#         flag_recogniser()
