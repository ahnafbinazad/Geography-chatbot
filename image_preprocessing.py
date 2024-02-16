import os
import numpy as np
from PIL import Image

# Define the directory path
flags_dir = 'flags'
data_dir = 'data'

# Function to load and preprocess images
def load_images_from_directory(directory):
    images = []
    labels = []
    for country_dir in os.listdir(directory):
        country_path = os.path.join(directory, country_dir)
        if os.path.isdir(country_path):
            print(f"Processing images for country: {country_dir}")
            for filename in os.listdir(country_path):
                image_path = os.path.join(country_path, filename)
                try:
                    # Load image and preprocess if needed
                    with Image.open(image_path) as image:
                        # Resize, convert to grayscale, or perform other preprocessing as necessary
                        image = image.resize((28, 28))  # Resize image to (28, 28)
                        image = np.array(image) / 255.0  # Normalize pixel values
                        images.append(image)
                        labels.append(country_dir)  # Assuming directory name is the label
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)


# Load train and test images along with actual country names as labels
train_images, train_labels = load_images_from_directory(os.path.join(flags_dir, 'train'))
test_images, test_labels = load_images_from_directory(os.path.join(flags_dir, 'test'))

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

# Save processed images and labels
np.save(os.path.join(data_dir, 'train_images.npy'), train_images)
np.save(os.path.join(data_dir, 'test_images.npy'), test_images)
np.save(os.path.join(data_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(data_dir, 'test_labels.npy'), test_labels)

print("Data saved successfully.")
