from PIL import Image
import os

jpg_directory = r"C:\Users\MikeBurgess\PycharmProjects\PythonProject\inputs\jpg_images"
width = 640
height = 640
tiles_directory = r"C:\Users\MikeBurgess\PycharmProjects\PythonProject\outputs\jpg_tiles"

if not os.path.exists(tiles_directory):
    os.makedirs(tiles_directory)

def create_tiles(img, width, height):
    img_width, img_height = img.size
    tiles = []

    for x in range(0,img_width,width):
        for y in range(0,img_height, height):
            box = (x, y, x+width, y+height)
            tile = img.crop(box)

            if tile.size != (width,height):
                # Resize tile to the desired size if it's too small
                tile = tile.resize((width,height))

            tiles.append(tile)

    return tiles


for filename in os.listdir(jpg_directory):
    if filename.endswith(".jpg") or filename.endsWith(".jpeg"):
        img_path = os.path.join(jpg_directory, filename)
        img = Image.open(img_path)
        tiles = create_tiles(img, width, height)

        # Save each tile
        for i, tile in enumerate(tiles):
            tile_path = os.path.join(tiles_directory, f"{os.path.splitext(filename)[0]}_tile_{i}.jpg")
            tile.save(tile_path)