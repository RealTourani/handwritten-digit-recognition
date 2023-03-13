import argparse
import os
import tensorflow as tf
import numpy as np
from PIL import Image


# Define number to label mapping
number_map = {
    0: 1,
    1: 2,
    2: 3,
    3: 4
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Detect a number in an image')
parser.add_argument('-img', dest='image_path', help='path to the image')
args = parser.parse_args()

# Check if image path is valid
if args.image_path is None or not os.path.exists(args.image_path):
    print('Error: Invalid image path')
else:
    # Load the trained model
    my_model = tf.keras.models.load_model('trained_model.h5')

    # Load the image using PIL
    new_image = Image.open(args.image_path)

    # Resize the image to 64x64
    new_image = new_image.resize((64, 64))

    # Convert the image to a numpy array
    new_image = np.array(new_image)

    # Add a fourth dimension to the array to represent the batch size (needed for model.predict)
    new_image = np.expand_dims(new_image, axis=0)

    # Make the prediction using the trained model
    prediction = my_model.predict(new_image)

    # Get the index of the predicted class
    predicted_class_index = np.argmax(prediction)

    # Map the predicted class index to a number
    if predicted_class_index in number_map:
        predicted_number = number_map[predicted_class_index]
        print(f"The predicted number is: {predicted_number}")
    else:
        print("Error: Invalid prediction")
