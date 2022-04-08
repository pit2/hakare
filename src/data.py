import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH_TO_DATA = "/Volumes/MACBACKUP/DataSets/ETL-9-examples.hdf5"


class Characters(Dataset):
    """Dataset structure for labeled kanji, hiragana, katakana  images."""

    def __init__(self, path, width, height, channels):
        """Create new dataset."""
        print("beginning file load")
        with h5py.File(path, "r") as file:
            keys = list(file.keys())
            img_key = keys[0]  # image data
            self.img_data = torch.tensor(list(file[img_key]))
            print("Loaded image data")
            label_key = keys[1]
            self.label_data = torch.tensor(list(file[label_key]))
            print("file loading complete")

        self.height = height
        self.width = width
        self.channels = channels

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
    test_generator = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_generator, valid_generator, test_generator


def dummy():
    data = Characters(PATH_TO_DATA, 127, 128, 1)
    print(len(data))
    transform = transforms.ToPILImage()
    img = transform(data.img_data[3000].view(data.width, data.height))
    img.show()
