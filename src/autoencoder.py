import torch
import data
import numpy as np
from os import path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AutoEncoder(torch.nn.Module):
    def __init__(self, decode=True):
        """ torch.nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, output_padding=1),
            torch.nn.Sigmoid(),
            torch.nn.Upsample(scale_factor=2, mode="nearest")
        """
        super().__init__()

        self.decode = decode
        self.encoderCNN = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            # torch.nn.Linear(4064, 2032), torch.nn.Linear(2032, 1016), torch.nn.Linear(1016, 508),
            # torch.nn.Linear(508, 254)
        )

        self.encoderFC = torch.nn.Sequential(
            torch.nn.Linear(128*2*2, 25),
            torch.nn.ReLU()
        )

        self.decoderCNN = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),

            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),

            torch.nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),


        )

        self.decoderFC = torch.nn.Sequential(
            torch.nn.Linear(25, 128*2*2),
            torch.nn.ReLU()
        )
    # with pool padding: 128 * 3 * 3

    def forward(self, x):
        x = self.encoderCNN(x)
        # print(f"Encoded shape: {x.shape}")
        x = x.view(-1, 128*2*2)
        x = self.encoderFC(x)
        # print(f"Shape after FC encoding: {x.shape}")
        if self.decode:
            x = self.decoderFC(x)
            # print(f"Shape after FC decoding: {x.shape}")
            x = x.view(-1, 128, 2, 2)
            # print(f"Shape for decoder CNN: {x.shape}")
            x = self.decoderCNN(x)
            # print(f"Shape after decoding: {x.shape}")
        return x


def train(model, training_gen, valid_gen, epochs):
    mse = torch.nn.MSELoss()
    adam = torch.optim.Adam(
        model.parameters(), lr=1e-1, weight_decay=1e-8)
    train_losses = []
    valid_losses = []
    min_valid_loss = np.inf
    outputs = []
    for i in range(epochs):
        model, train_losses, outputs = step(
            model, training_gen, mse, adam, train_losses, outputs)
        print(f"Error in epoch {i}: {train_losses[-1]}")
        valid_losses = eval_model(model, valid_gen, mse, valid_losses)
        if min_valid_loss > valid_losses[-1]:
            torch.save(model, path.join("..", "data", "models", "model.pt"))
    return model, train_losses, valid_losses


def step(model, generator, loss_function, optimizer, losses, outputs):
    model.train()
    first = True
    local_loss = []
    for imgs, _ in generator:
        #    print(torch.cuda.memory_allocated()/1024**2)
        imgs = imgs.to(DEVICE)

        decoded = model(imgs)
        loss = loss_function(imgs, decoded)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        local_loss.append(loss.item())
        if first:
            outputs.append((imgs, decoded))
        first = False
    err = np.mean(local_loss)
    losses.append(err)
    torch.cuda.empty_cache()
    return model, losses, outputs


def eval_model(model, generator, loss_function, losses):
    model.eval()
    local_errors = []
    with torch.no_grad():
        for imgs, _ in generator:
            decoded = model(imgs)
            loss = loss_function(imgs, decoded)
            local_errors.append(loss.item())
        losses.append(np.mean(local_errors))

    return losses


def reconstruct(img_path, model):
    img_tensor = _img_to_tensor(img_path)

    model.eval()
    reconstructed_tensor = model(img_tensor)
    reconstructed_tensor = reconstructed_tensor.view(1, 128, 128)
    save_image(img_tensor, path.join("..", "data", "tmp", "back.png"))
    print(img_tensor - reconstructed_tensor)

    transform = transforms.ToPILImage()
    reconstructed = transform(reconstructed_tensor)
    return reconstructed


def encode(img_path, model):

    img_tensor = _img_to_tensor(img_path)
    img_tensor = img_tensor.view(1, 1, 128, 128)
    model.decode = False
    model.eval()
    encoded_tensor = model(img_tensor)
    return encoded_tensor


def _img_to_tensor(img_path):
    img = Image.open(img_path)
    img = img.resize((128, 128))
    transform = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform(img)
    img_tensor = img_tensor / 255

    return img_tensor


def execute():
    dataset = data.Characters(data.PATH_TO_DATA_SHORT, 128, 128, 1)
    X_train, _, _ = data.split(dataset)

    train(X_train, 10)
