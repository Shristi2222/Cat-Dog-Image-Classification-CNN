import os
from PIL import Image

DATASET_PATH = "../dataset/dogs-vs-cats-classification"

bad_files = []

print("Scanning images...")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(root, file)

            try:

                img = Image.open(path)

                img.verify()

            except Exception:

                bad_files.append(path)

print("\nFinished scanning.\n")

print("Corrupted Images Found:", len(bad_files))

for file in bad_files:

    print(file)