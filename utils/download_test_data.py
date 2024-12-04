import json
import random
import os
from PIL import Image, ImageDraw

# Function to create synthetic images and annotations
def create_synthetic_coco_dataset(output_dir='dataset', num_images=10, image_size=(600, 600), num_objects_per_image=5):
    categories = [{"id": 1, "name": "circle", "supercategory": "shape"},
                  {"id": 2, "name": "rectangle", "supercategory": "shape"},
                  {"id": 3, "name": "triangle", "supercategory": "shape"}]  # Added third category

    images = []
    annotations = []
    annotation_id = 1  # Start annotation IDs from 1

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create images and annotations
    for img_id in range(1, num_images + 1):
        img_name = f"image_{img_id}.jpg"
        img_path = os.path.join(output_dir, img_name)

        images.append({
            "id": img_id,
            "file_name": img_name,  # Just the image filename, not the path
            "width": image_size[0],
            "height": image_size[1]
        })

        # Create the image with a white background
        img = Image.new('RGB', image_size, (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw a random number of objects (rectangles, circles, or triangles)
        for obj_id in range(1, num_objects_per_image + 1):
            x_min = random.randint(0, image_size[0] - 50)
            y_min = random.randint(0, image_size[1] - 50)
            width = random.randint(50, 100)
            height = random.randint(50, 100)

            obj_category = random.choice(categories)
            annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": obj_category["id"],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [],  # Empty segmentation, can be added if needed
            })
            annotation_id += 1

            # Draw the object (shape)
            if obj_category["name"] == "circle":
                draw.ellipse([x_min, y_min, x_min + width, y_min + height], fill=(255, 0, 0))
            elif obj_category["name"] == "rectangle":
                draw.rectangle([x_min, y_min, x_min + width, y_min + height], fill=(0, 255, 0))
            elif obj_category["name"] == "triangle":
                draw.polygon([
                    (x_min + width // 2, y_min),
                    (x_min, y_min + height),
                    (x_min + width, y_min + height)
                ], fill=(0, 0, 255))

        img.save(img_path)

    # Create COCO format for the entire dataset
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save the annotations in the same folder
    annotations_file = os.path.join(output_dir, 'annotations.json')
    with open(annotations_file, "w") as f:
        json.dump(coco_format, f)

# Example usage: Generate a synthetic dataset with images and annotations in 'charlixcx' folder
create_synthetic_coco_dataset(output_dir='dataset/data/train', num_images=8, num_objects_per_image=3)
create_synthetic_coco_dataset(output_dir='dataset/data/val', num_images=1, num_objects_per_image=3)
create_synthetic_coco_dataset(output_dir='dataset/data/test', num_images=1, num_objects_per_image=3)
