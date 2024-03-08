import os
import numpy as np
from PIL import Image
from tensorflow import keras
import contextlib

from text_to_speech import text_to_speech
from class_names import countries

class FlagRecogniser:
    def __init__(self):
        # Initialize the FlagRecogniser class by loading the trained model and compiling it
        self.model = self.load_model()
        self.compile_model(self.model)

    # Function to preprocess the image before passing it to the model
    @staticmethod
    def preprocess_image(image_path):
        """
        Preprocesses the input image.

        Args:
            image_path (str): The path to the input image file.

        Returns:
            np.ndarray: Preprocessed image array.
        """
        img = Image.open(image_path)
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    # Function to load the trained model
    @staticmethod
    def load_model():
        """
        Loads the trained model from the specified path.

        Returns:
            keras.Model: Loaded Keras model.
        """
        model_path = 'model/trained_model.h5'
        model = keras.models.load_model(model_path)
        return model

    # Function to compile the loaded model
    @staticmethod
    def compile_model(model):
        """
        Compiles the given Keras model with specified optimizer, loss function, and metrics.

        Args:
            model (keras.Model): The Keras model to be compiled.
        """
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Function to classify the image using the loaded model
    @staticmethod
    def classify_image(image_path, model, class_names):
        """
        Classifies the input image using the provided model.

        Args:
            image_path (str): The path to the input image file.
            model (keras.Model): The loaded Keras model for image classification.
            class_names (list): List of class names for mapping predicted indices to names.

        Returns:
            str: Predicted class name.
        """
        img_array = FlagRecogniser.preprocess_image(image_path)

        # Suppress print output during model prediction
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            prediction = model.predict(img_array)

        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        return predicted_class_name

    def flag_recogniser(self, voice_enabled):
        """
        Recognises the flag from the provided image file path and prints the predicted class name.

        Args:
            voice_enabled (bool): Specifies whether to enable text-to-speech functionality.
        """
        # Ask the user for a file path
        file_path = input("Please enter the file path of the image: ")
        abs_file_path = os.path.abspath(file_path)

        # Check if the file exists
        if not os.path.exists(abs_file_path):
            print("File does not exist.")
            return

        # Classify the image
        predicted_class_name = self.classify_image(file_path, self.model, countries)

        # Print the predicted class
        output = f"That is the flag of {predicted_class_name}"
        print(output)
        text_to_speech(voice_enabled, output)

if __name__ == "__main__":
    # Instantiate the FlagRecogniser class and continuously prompt for image input
    flag_recogniser = FlagRecogniser()
    while True:
        flag_recogniser.flag_recogniser(voice_enabled=False)
