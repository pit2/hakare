import torch
import h5py
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Characters(Dataset):
    def __init__(self, filename):
        with h5py.File(filename, "r") as file:
            keys = list(file.keys())
            img_key = keys[0]  # image data
            self.img_data = torch.tensor(list(file[img_key]))
            label_key = keys[1]
            self.label_data = torch.tensor(list(file[label_key]))

    def __len__(self):
        return len(self.img_data)

    def __index__(self, index):
        return self.img_data[index], self.label_data[index]



