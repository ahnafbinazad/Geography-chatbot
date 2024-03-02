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

    # Function to load the trained model
    @staticmethod
    def load_model():
        model_path = 'model/trained_model.h5'
        model = keras.models.load_model(model_path)
        return model

    # Function to classify the image
    def classify_image(self, image):
        img_array = FlagRecogniser.preprocess_image(image)

        # Suppress print output during model prediction
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            prediction = self.model.predict(img_array)

        predicted_class_index = np.argmax(prediction)
        predicted_class_name = countries[predicted_class_index]
        return predicted_class_name

    def process_video(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return

        # Dictionary to store count of recognized countries
        country_counts = {}

        # Process each frame
        frame_number = 0
        print("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as image
            image_path = f"frames/frame_{frame_number}.jpg"
            cv2.imwrite(image_path, frame)

            # Classify the frame
            predicted_class_name = self.classify_image(image_path)

            # Add recognised country text on top of the frame
            cv2.putText(frame, predicted_class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame with recognised country
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)  # Adjust the delay as needed

            # Count the recognized country
            if predicted_class_name in country_counts:
                country_counts[predicted_class_name] += 1
            else:
                country_counts[predicted_class_name] = 1

            frame_number += 1

        # Release video capture object
        cap.release()
        cv2.destroyAllWindows()

        # Find the most frequent country name
        most_frequent_country = max(country_counts, key=country_counts.get)
        print(f"This video contains the flag of {most_frequent_country}")

    def recognise_video(self):
        video_path = input("Please enter the file path of the MP4 video: ")
        self.process_video(video_path)


if __name__ == "__main__":
    videoRecognizer = VideoFlagRecognizer()
    videoRecognizer.recognise_video()
