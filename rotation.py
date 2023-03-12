import os
from PIL import Image

# Path to directory containing images
image_dir = 'QR_d_best/'

# Loop through all files in the directory
for file_name in os.listdir(image_dir):
    # Check if file is an image
    if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
        # Open the image
        img = Image.open(image_dir + file_name)
        # Rotate the image by 180 degrees
        rotated_img = img.rotate(180)
        # Save the rotated image over the original image
        rotated_img.save(image_dir + file_name)
