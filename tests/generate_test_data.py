# tests/generate_test_data.py
import os
from PIL import Image, ImageDraw

def ensure_test_data(root="tests/data", img_size=(64,64)):
    os.makedirs(root, exist_ok=True)

    # Define classes and colors
    classes = {
        "cat": (255, 160, 122),   # light salmon
        "dog": (160, 222, 120),   # light green
    }

    # For each class, create 2 images: <class>_0.jpg, <class>_1.jpg
    for cls, color in classes.items():
        for i in range(2):
            fname = f"{cls}_{i}.jpg"
            path = os.path.join(root, fname)
            if not os.path.exists(path):
                img = Image.new("RGB", img_size, color=color)
                # draw a simple shape to vary images a bit
                draw = ImageDraw.Draw(img)
                if cls == "cat":
                    draw.ellipse([8,8,img_size[0]-8,img_size[1]-8], fill=(255,255,255,0))
                else:
                    draw.rectangle([8,8,img_size[0]-8,img_size[1]-8], outline=(0,0,0))
                img.save(path, "JPEG", quality=85)

    # Write class_map.json consistent with data._get_image_files_and_labels
    # data._get_image_files_and_labels extracts 'cat' from 'cat_0.jpg'
    class_map = sorted(list(classes.keys()))
    with open(os.path.join(root, "class_map.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(class_map, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ensure_test_data()
    print("Generated test dataset at tests/data/")
