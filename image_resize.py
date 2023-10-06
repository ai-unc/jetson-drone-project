from PIL import Image
import os

# Specify the input and output directories
input_dir = "/Users/atin/Documents/GitHub/jetson-drone-project/binary_dataset/validation/negatives"
output_dir = "/Users/atin/Documents/GitHub/jetson-drone-project/resized_dataset/validation/negatives"

# Specify the target size (close to 256x256)
target_size = (224, 224)

# Iterate through the files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Open the image
        img = Image.open(os.path.join(input_dir, filename))

        # Calculate the new size while preserving the aspect ratio
        img.thumbnail(target_size)

        # Save the resized image to the output directory
        img.save(os.path.join(output_dir, filename))
