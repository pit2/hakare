from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os.path
import csv
import numpy as np


def img_test():
    """Opens an image, transforms it into tensor, and saves the tensor back as image file."""

    image = Image.open("data/images/ETL1/0x003d/059245.png")

    image.show()

    transform = transforms.Compose([transforms.PILToTensor()])

    img_tensor = transform(image)

    img_tensor = img_tensor / 255

    print(img_tensor)

    save_image(img_tensor, "data/tmp/back.png")


def convert_img_to_csv(directory, img_name, path_to_csv, label_list=[]):
    """Convert image to flattened nparray and write it in a row of csv file.

    Each row contains width*height columns for the actual image data, and one column for the label
    """

    image = Image.open(os.path.join(directory, img_name))
    row = np.array(image).flatten()
    with open(os.path.join(directory, ".char.txt"), "r") as txt_file:
        label = txt_file.read()
    label_list.append(label)
    row = np.append(row, len(label_list) - 1)
    with open(path_to_csv, "a") as file:
        writer = csv.writer(file)
        writer.writerow(row)


convert_img_to_csv("data/images/ETL1/0x003d", "059245.png", "data/test.csv")
