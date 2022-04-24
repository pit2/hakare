import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math

PATH_TO_DATA = "/Volumes/MACBACKUP/DataSets/ETL-9-128x128.hdf5"
PATH_TO_DATA_SHORT = "/Volumes/MACBACKUP/DataSets/ETL-9-1000.hdf5"
PATH_TO_DATA_MINI = "/Volumes/MACBACKUP/DataSets/ETL-9-128x128-10.hdf5"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Characters(Dataset):
    def __init__(self, path, num_chunks, width, height, channels):
        self.path = path
        self.cache_idx = []
        self.img_data = None
        self.label_data = None
        self.file = None
        self.width = width
        self.height = height
        self.channels = channels
        self.chunks = num_chunks
        self.mean = np.nan
        self.std = np.nan
        with h5py.File(self.path, "r") as file:
            self.size = (int)(len(file["images"]))
        self.chunk_size = (int)(math.floor(self.size / self.chunks))

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
            self.img_data = self.file["images"]
            self.label_data = self.file["labels"]

        return self.img_data[index], self.label_data[index]

    def __len__(self):
        return self.size


def transform(imgs, labels, mean, std, channels=1, width=128, height=128):
    """Transform slice of img/label hdf5 file into tensor,
    reshaped into (-1, channels, width, height).

    Parameters:
    - imgs (img group of hdf5 file) - images group in hdf5 file.
    - labels (label group of hdf5 file) - labels group in hdf5 file.
    - channels (int) - number of channels for the image data.
    - width (int) - width of image data.
    - height (int) - height of image data.

    Returns
    - image tensor of shape (-1, channels, width, height), where the first dimension is
        the batch size.
    - label tensor of shape (-1, 1) (squeezed).
    """

    imgs = imgs / 255
    transform = transforms.Normalize(mean, std)
    imgs = imgs.view(-1, channels, width, height)
    imgs = transform(imgs)
    imgs = imgs.view(-1, channels, width, height)

    labels = labels.squeeze()
    return imgs, labels


def get_mean_std(iter):
    count = 0
    mean = torch.empty(1)
    std_aux = torch.empty(1)
    for imgs, _ in iter:
        imgs = imgs / 255
        mean, std_aux, count = get_mean_std_(imgs, count, mean, std_aux)

    return mean, torch.sqrt((std_aux - mean) ** 2)


def get_mean_std_(imgs, count, mean, std_aux):

    imgs = imgs.view(-1, 128 * 128)
    _, num_of_px = imgs.shape
    num = count + num_of_px
    sum_ = torch.sum(imgs)
    sum_of_squares = torch.sum(imgs ** 2)

    mean = (count * mean + sum_) / num
    std_aux = (count * std_aux + sum_of_squares) / num
    count += num_of_px

    return mean, std_aux, count


def split(data, batch_size=64, train=0.5, valid=0.2, num_workers=0):
    """Split data into train/valid/test sets of given batch size. Returns iterable generators."""
    train_size = int(train * len(data))
    valid_size = int(valid * len(data))
    test_size = len(data) - train_size - valid_size
    train_data, valid_data, test_data = torch.utils.data.random_split(
        data, (train_size, valid_size, test_size))

    train_generator = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_generator = DataLoader(
        valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_generator = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_generator, valid_generator, test_generator


def dummy():
    data = Characters(PATH_TO_DATA_SHORT, 128, 128, 1)
    print(len(data))
    print(f"data shape in dummy: {data[3000][0].shape}")
    transform = transforms.ToPILImage()
    img = transform(data[3000][0].view(data.width, data.height))
    img.show()
    img = transform(data[2499][0].view(data.width, data.height))
    img.show()


# data = Characters(PATH_TO_DATA, 127, 128, 1)
# dummy()
