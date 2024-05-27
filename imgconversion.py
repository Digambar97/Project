import os
from PIL import Image

# jpg_folder = "displayed_images/images/images"
png_folder = "displayed_images/images/png_images"
min_width, min_height = 440, 581

if not os.path.exists(png_folder):
    os.makedirs(png_folder)

for filename in os.listdir(png_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(png_folder, filename)
        img = Image.open(image_path)

        if img.width < min_width or img.height < min_height:
            img = img.resize((min_width, min_height), Image.LANCZOS)

        # img.save(os.path.join(png_folder, filename[:-4] + ".png"), "PNG")

print("Conversion completed!")