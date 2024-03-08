import cv2
import os
import numpy as np
from tensorflow import keras
import contextlib
from flag_recogniser import FlagRecogniser
from class_names import countries


class VideoFlagRecognizer:
    def __init__(self):
        self.model = self.load_model()

    @staticmethod
    def load_model():
        """
        Loads the trained model.

        Returns:
        - model: Loaded Keras model.
        """
        model_path = 'model/trained_model.h5'
        model = keras.models.load_model(model_path)
        return model

    def classify_image(self, image):
        """
        Classifies an image using the loaded model.

        Parameters:
        - image: String, path to the image file.

        Returns:
        - predicted_class_name: String, name of the predicted country.
        """
        img_array = FlagRecogniser.preprocess_image(image)

        # Suppress print output during model prediction
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            prediction = self.model.predict(img_array)

        predicted_class_index = np.argmax(prediction)
        predicted_class_name = countries[predicted_class_index]
        return predicted_class_name

    def process_video(self, video_path):
        """
        Processes a video file, recognizing flags in each frame.

        Parameters:
        - video_path: String, path to the video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return

        country_counts = {}
        frame_number = 0
        print("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_path = f"frames/frame_{frame_number}.jpg"
            cv2.imwrite(image_path, frame)

            predicted_class_name = self.classify_image(image_path)

            cv2.putText(frame, predicted_class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

            if predicted_class_name in country_counts:
                country_counts[predicted_class_name] += 1
            else:
                country_counts[predicted_class_name] = 1

            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

        most_frequent_country = max(country_counts, key=country_counts.get)
        print(f"This video contains the flag of {most_frequent_country}")

    def recognise_video(self):
        """
        Prompts the user to enter the file path of the video and initiates the video processing.
        """
        video_path = input("Please enter the file path of the MP4 video: ")
        self.process_video(video_path)


if __name__ == "__main__":
    videoRecognizer = VideoFlagRecognizer()
    videoRecognizer.recognise_video()
