import os
import shutil
import random

SOURCE_DIR = "dataset"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

SPLIT_RATIO = 0.8

def create_folders(base_dir, classes):
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

def split_data():
    classes = os.listdir(SOURCE_DIR)

    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)

        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        split_index = int(len(images) * SPLIT_RATIO)

        train_images = images[:split_index]
        val_images = images[split_index:]

        create_folders(TRAIN_DIR, classes)
        create_folders(VAL_DIR, classes)

        for img in train_images:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(TRAIN_DIR, cls, img))

        for img in val_images:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(VAL_DIR, cls, img))

        print(f"{cls} → Train: {len(train_images)}, Val: {len(val_images)}")

    print("✅ Done!")

if __name__ == "__main__":
    split_data()