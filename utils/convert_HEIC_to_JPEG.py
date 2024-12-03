import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def convert_heic_to_jpg(heic_path, jpg_path):
    image = Image.open(heic_path)
    image.save(jpg_path, format="JPEG")

heic_directory = r"C:\Users\MikeBurgess\Downloads\most_recent_images"
jpg_directory = r"C:\Users\MikeBurgess\PycharmProjects\PythonProject\inputs\jpg_images"

if not os.path.exists(jpg_directory):
    os.makedirs(jpg_directory)

for filename in os.listdir(heic_directory):
    if filename.endswith(".heic") or filename.endswith(".HEIC"):
        heic_path = os.path.join(heic_directory, filename)
        filename_without_extension = os.path.splitext(filename)[0]
        jpg_path = os.path.join(jpg_directory, filename_without_extension + ".jpg")
        convert_heic_to_jpg(heic_path, jpg_path)