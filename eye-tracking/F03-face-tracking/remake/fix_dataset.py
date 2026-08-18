#!/usr/bin/env python3
"""
Reorganize CASIA-WebFace images into ImageFolder format.

Before:
    dataset/
        0000045_001.jpg
        0000045_002.jpg
        0000099_001.jpg
        ...

After:
    dataset/
        0000045/
            001.jpg
            002.jpg
        0000099/
            001.jpg
"""

from pathlib import Path
import shutil

# Path to the dataset
DATASET_DIR = Path(r"F03-face-tracking\remake\data\faces_webface_112x112\CASIA-WebFace_crop")   # <-- Change this


def main():

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    count = 0

    for file in DATASET_DIR.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in image_extensions:
            continue

        stem = file.stem

        # Expecting something like 0000045_001
        try:
            identity, image_number = stem.split("_", 1)
        except ValueError:
            print(f"Skipping {file.name}")
            continue

        # Create identity folder
        identity_folder = DATASET_DIR / identity
        identity_folder.mkdir(exist_ok=True)

        # New filename
        new_name = image_number + file.suffix.lower()

        destination = identity_folder / new_name

        shutil.move(str(file), str(destination))

        count += 1

        if count % 1000 == 0:
            print(f"{count} images processed...")

    print(f"\nDone! {count} images moved.")


if __name__ == "__main__":
    main()