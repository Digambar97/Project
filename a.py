import cv2
import numpy as np
import os

# Define the input and output folders
input_folder = 'displayed_images\images\images'
output_folder = 'displayed_images\images\png_images'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # Check for JPEG images
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to create a mask where the background is white
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Invert the mask to get the background as black
        mask_inv = cv2.bitwise_not(mask)

        # Convert the original image to BGRA (Blue, Green, Red, Alpha)
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Split the channels
        b, g, r, a = cv2.split(image_bgra)

        # Use the inverted mask to set the alpha channel of the image
        a = cv2.bitwise_and(mask_inv, a)

        # Merge the channels back
        image_bgra = cv2.merge((b, g, r, a))

        # Save the result
        output_path = os.path.join(output_folder, filename[:-4] + '.png')
        cv2.imwrite(output_path, image_bgra)
