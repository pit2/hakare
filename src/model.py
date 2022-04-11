import torch
import data
import numpy as np
from os import path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchsummary import summary
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Recognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            #
        )
        self.fc = torch.nn.Linear(128 * 4 * 4, 3036)
        self.train_losses = []
        self.valid_losses = []
        self.test_loss = np.inf
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 128 * 4 * 4)
        return self.fc(x)


def execute():
    model = Recognizer().to(DEVICE)
    summary(model, (1, 128, 128))

    dataset = data.Characters(data.PATH_TO_DATA_SHORT, 128, 128, 1)
    train, valid, test = data.split(dataset, batch_size=32)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-1, weight_decay=1e-8)
    for i in range(25):

        local_loss = []
        for imgs, targets in train:
            #    print(torch.cuda.memory_allocated()/1024**2)
            imgs = imgs.to(DEVICE)

            predicts = model(imgs)
            loss = model.loss_fn(predicts, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            local_loss.append(loss.item())

        err = np.mean(local_loss)
        model.train_losses.append(err)
        print(f"Error in epoch {i}: {err}")
        torch.cuda.empty_cache()

    plt.plot(model.train_losses)


execute()
