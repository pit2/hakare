import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os.path

PATH_TO_DATA = os.path.join("data", "ETL-9-90x90.hdf5")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Characters(Dataset):
    """Characters data class that lazily loads hdf5 file containing serialised images and
    labels.

    Parameters:
        path (str): Path to hdf5 data file.
        channels (int): Number of channels per image.
        width (int): Width of each image.
        height (int): Height of each image.
    """

    def __init__(self, path, channels, width, height):

        self.path = path
        self.img_data = None
        self.label_data = None
        self.file = None
        self.width = width
        self.height = height
        self.channels = channels
        self.mean = np.nan
        self.std = np.nan
        with h5py.File(self.path, "r") as file:
            self.size = (int)(len(file["images"]))

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
            self.img_data = self.file["images"]
            self.label_data = self.file["labels"]

        return self.img_data[index], self.label_data[index]

    def __len__(self):
        return self.size


def transform(imgs, labels, mean, std, channels=1, width=90, height=90):
    """Transform slice of img/label hdf5 file into tensor,
    reshaped into normalised float tensor of shape (-1, channels, width, height).

    Parameters:
    - imgs (img group of hdf5 file) - images group in hdf5 file.
    - labels (label group of hdf5 file) - labels group in hdf5 file.
    - mean (float): Mean to use for normalisation.
    - std (float): Standard deviation to use for normalisation.
    - channels (int) - number of channels for the image data.
    - width (int) - width of image data.
    - height (int) - height of image data.

    Returns
    - image float tensor of shape (-1, channels, width, height), where the first dimension is
        the batch size.
    - label long tensor of shape (-1, 1) (squeezed).
    """

    imgs = imgs.view(-1, channels, width, height)
    imgs = imgs.float()
    transform = transforms.Normalize(mean=mean, std=std)
    imgs = imgs / 255
    imgs = transform(imgs)
    labels = labels.long().squeeze()
    return imgs, labels


def get_mean_std(iter):
    """Returns mean, std of iter (Dataloader)."""
    means = []
    stds = []
    for imgs, _ in iter:
        imgs = imgs.view(-1, 90 * 90)
        imgs = imgs / 255
        mean = torch.mean(imgs)
        std = torch.std(imgs)
        means.append(mean)
        stds.append(std)

    return np.array(means).mean(axis=0), np.array(stds).mean(axis=0)


def split(data, batch_size=64, train=0.5, valid=0.2, test=0.3, num_workers=0):
    """Randomly split data into train/valid/test sets of given batch size. Returns iterable generators.
    If the proportions of train, valid and test sum up to less than one, the remaing data
    is discarded.

    Parameters:
    data (Characters): Instance of characters class containing the entire data set.
    batch_size (int): Batch size to use
    train (float in intervall (0, 1)): data to be used for training set
    valid (float in interval (0, 1)): data to be used for validation set
    test (float in interval (0,1)): data to be used for test set
    num_workers (int): Number of workers for Dataloader

    Returns
        train_generator (Dataloader)
        valid_generator (Dataloader)
        test_generator (Dataloader)
    """
    train_size = int(train * len(data))
    valid_size = int(valid * len(data))
    test_size = int(test * len(data))
    rest = len(data) - train_size - valid_size - test_size
    train_data, valid_data, test_data, _ = torch.utils.data.random_split(
        data, (train_size, valid_size, test_size, rest))

    train_generator = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_generator = DataLoader(
        valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_generator = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_generator, valid_generator, test_generator
