import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PATH_TO_DATA = "/Volumes/MACBACKUP/DataSets/ETL-9.hdf5"
PATH_TO_DATA_SHORT = "/Volumes/MACBACKUP/DataSets/ETL-9-1000.hdf5"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Characters(Dataset):
    def __init__(self, path, width, height, channels):
        self.path = path
        self.img_data = None
        self.label_data = None
        self.width = width
        self.height = height
        self.channels = channels
        with h5py.File(self.path, "r") as file:
            self.size = len(file["images"])

    def __getitem__(self, index):
        if self.img_data is None:
            with h5py.File(self.path, "r") as file:
                self.img_data = torch.tensor(np.array(file["images"]))
                self.img_data = self.img_data / 255
                self.img_data = self.img_data.view(-1,
                                                   self.channels, self. width, self.height)
                self.img_data.to(DEVICE)
                # print(f"Data shape in getitem: {self.img_data.shape}")
                self.label_data = torch.tensor(
                    np.array(file["labels"])).squeeze().to(DEVICE)
        return self.img_data[index], self.label_data[index]

    def __len__(self):
        return self.size


class CharactersWithChache(Dataset):
    """Dataset structure for labeled kanji, hiragana, katakana  images.
    Unfinished - implements manual caching
    """

    def __init__(self, path, width, height, channels, data_cache_size=3):
        """Create new dataset."""

        self.data_info = []
        self.data_cache = []
        self.data_cache_size = data_cache_size
        self._add_data_infos(path)
        """with h5py.File(path, "r") as file:
            keys = list(file.keys())
            img_key = keys[0]  # image data
            self.img_data = torch.tensor(list(file[img_key]))
            print("Loaded image data")
            label_key = keys[1]
            self.label_data = torch.tensor(list(file[label_key]))
            print("file loading complete")"""

        self.height = height
        self.width = width
        self.channels = channels

    def _add_data_infos(self, path):
        print_first = True
        with h5py.File(path, "r") as file:
            for gname, group in file.items():
                for data in group:
                    idx = -1
                    self.data_info.append(
                        {"type": data.dtype, "shape": data.shape, "cache_idx": idx})
                    if print_first:
                        print_first = False

    def _add_to_cache(self, data):
        self.data_cache.append(data)
        return len(self.data_cache) - 1

    def _load_data(self, path):
        with h5py.File(path, "r") as file:
            for gname, group in file.items():
                for data in group:
                    cache_idx = self._add_to_cache(data)
                    file_idx = next(i for i, v in enumerate(self.data_info))
                    self.data_info[file_idx +
                                   cache_idx]["cache_idx"] = cache_idx

        print(f"Removal keys: {list(self.data_cache)}")
        if len(self.data_cache) > self.data_cache_size:
            pass

    def __len__(self):
        """Returns # of images in data set."""
        return len(self.img_data)

    def __getitem__(self, index):
        """Returns tuple (image, label) at given index."""
        return self.img_data[index], self.label_data[index]


def split(data, batch_size=64, train=0.5, valid=0.2):
    train_size = int(train * len(data))
    valid_size = int(valid * len(data))
    test_size = len(data) - train_size - valid_size
    train_data, valid_data, test_data = torch.utils.data.random_split(
        data, (train_size, valid_size, test_size))

    train_generator = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    valid_generator = DataLoader(
        valid_data, batch_size=batch_size, shuffle=True)
    test_generator = DataLoader(
        test_data, batch_size=batch_size, shuffle=True)

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
