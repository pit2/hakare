from PIL import Image
import os.path
import csv
import numpy as np
import h5py


def convert_img_to_array(directory, img_name, labels=[], resize=(128, 128), crop=(90, 90)):
    """Convert image to flattened nparray.

    Each row contains width*height columns for the actual image data.
    This method assumes the label
    is the sole content of a file .char.text located the same directory as the image.

    Parameters:
        directory (string): Path to the directory where the image is located.
        img_name (string): The file name of the image, including extension.
        labels (list of strings): Duplicate-free list of strings of previously read labels.
        resize (int, int): Dimension to resize the image to before cropping.
        crop (int, int): Dimension to center-crop the image to.

    Returns:
        row (nparray) - image converted to 1D array.
        ind (int) - index of label
        labels parameter (list of strings) augmented by the newly-added label.
    """

    image = Image.open(os.path.join(directory, img_name))
    image = image.resize(resize)
    width, height = image.size
    image = image.crop(((width - crop[0]) / 2, (height - crop[1]) / 2,
                       (width + crop[0]) / 2, (height + crop[1]) / 2))

    row = np.array(image).flatten()
    with open(os.path.join(directory, ".char.txt"), "r") as txt_file:
        label = txt_file.read()
    if label not in labels:
        labels.append(label)
        ind = len(labels) - 1
    else:
        ind = labels.index(label)

    return row, ind, labels


def convert_to_csv(directory, path_to_csv, path_to_labels_list, labels=[], limit=0):
    """Read directory e.g. ETL9 and scan its subdirectories for images.

    Each subdirectory must contain a collection of images and a .char.txt file with the label as
    its sole content. The method writes the images that have been converted to arrays to the
    given csv file, one line per image, augmented by the corresponding label in the last column.
    It also saves the labels list as csv (first column index, second is label).

    Parameters:
        directory (string): Root directory, e.g. data/images/ETL9, for the image files.
        path_to_csv (string): Path including file name of output csv file.
        path_to_labels_list (string): Path including file name of labels dictionary csv file.
        labels (list of strings): List of labels, default empty, but could be converted contents
            of labels csv dictionary as well.
        limit (int): number of images to be crawled, 0 to crawl all images
                     in the directory.
    """
    with open(path_to_csv, "a") as csv_file:
        count = 0
        for root, _, files in os.walk(directory):
            for file in [f for f in files if not f[0] == "."]:
                row, ind, labels = convert_img_to_array(root, file, labels)
                row = np.append(row, ind)
                writer = csv.writer(csv_file)
                writer.writerow(row)
                count += 1
                if count > limit > 0:
                    break

    _write_labels(path_to_labels_list, labels)


def _write_labels(path, labels):
    """Write labels to csv file in specified path."""
    with open(path, "w") as file:
        writer = csv.writer(file)
        for i in range(len(labels)):
            writer.writerow([i, labels[i]])


def convert_to_hdf5(directory, path_to_hdf5, path_to_labels_list, labels=[], limit=0):
    """Read directory e.g. ETL9 and scan its subdirectories for images. The images and
    corresponding labels are then saved to a hdf5 file.

    Each subdirectory must contain a collection of images and a .char.txt file with the label as
    its sole content. The method writes the images that have been converted to arrays to the
    given hdf5 file, with group key 'images', and group "labels" for the labels.
    It also saves the labels list as csv (first column index, second is label).

    Parameters:
        directory (string): Root directory, e.g. data/images/ETL9, for the image files.
        path_to_hdf5 (string): Path including file name of output hdf5 file.
        path_to_labels_list (string): Path including file name of labels dictionary csv file.
        labels (list of strings): List of labels, default empty, but could be converted contents
            of labels csv dictionary as well.
        limit (int): number of images to be crawled 0 to crawl all images
            in the directory.
    """
    count = 0
    img_rows = []
    labels_column = []
    for root, _, files in os.walk(directory):
        for file in [f for f in files if not f[0] == "."]:
            row, ind, labels = convert_img_to_array(root, file, labels)
            img_rows.append(row)
            labels_column.append(ind)
            count += 1
            if count > limit > 0:
                break

    with h5py.File(path_to_hdf5, "a") as hdf5:
        hdf5.create_dataset("images", data=np.array(img_rows))
        hdf5.create_dataset("labels", data=np.array(labels_column))

    _write_labels(path_to_labels_list, labels)


def read_hdf5(filename):
    """Test function to read and show image loaded from hdf5 file."""
    with h5py.File(filename, "r") as f:
        key = list(f.keys())[0]
        data = np.array(list(f[key]))
        img = data[30].reshape(90, 90)
        Image.fromarray(img).show()
        print(len(data))
