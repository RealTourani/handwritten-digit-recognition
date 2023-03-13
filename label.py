import csv
from pathlib import Path


def calculate_label(img_basename):
    path = img_basename
    img_basename = img_basename.rsplit("\\", 1)[1]  # Remove the path characters and keep only image name
    tmp = img_basename[:len(img_basename) - 4].split("_")
    row = tmp[2]
    rod_id = int(tmp[3])
    if row in {"A", "B", "E", "F", "I", "J"}:
        label = 4 - (rod_id % 4)
    else:
        label = (rod_id % 4) + 1

    return path, label


header = ['name', 'label']  # Define header for CSV file

images = Path("QR_d_best").glob("*.jpg")  # select all jpg file in mentioned path
image_strings = [str(p) for p in images]  # Read all images

with open('labels.csv', 'w', encoding="UTF8",
          newline='') as f:  # Open CSV file and write path and label of images into CSV file
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(header)  # Write headers into CSV file

    for i in image_strings:
        path, label = calculate_label(i)  # call calculate_label function and get path and label of images

        writer.writerow([path, label])  # Write path and labels into CSV file
