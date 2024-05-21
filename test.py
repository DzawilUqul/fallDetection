import os
from PIL import Image

# Define the folder containing images and the target size
input_folder = 'D:/EternityBee/UGM/Computer Vision/Person Detection/output'
output_folder = 'D:/EternityBee/UGM/Computer Vision/Person Detection/output_resized'
target_size = (56, 46)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Open the image file
        with Image.open(os.path.join(input_folder, filename)) as img:
            # Resize the image
            resized_img = img.resize(target_size, Image.LANCZOS)
            # Save the resized image to the output folder
            resized_img.save(os.path.join(output_folder, filename))

print("All images have been resized.")
