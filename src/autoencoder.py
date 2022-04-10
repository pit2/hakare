import torch
from torchsummary import summary
import data

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AutoEncoder(torch.nn.Module):
    def __init__(self):
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
            torch.nn.Linear(128*2*2, 10),
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
            torch.nn.Linear(10, 128*2*2),
            torch.nn.ReLU()
        )
    # with pool padding: 128 * 3 * 3

    def forward(self, x):
        x = self.encoderCNN(x)
        # print(f"Encoded shape: {x.shape}")
        x = x.view(-1, 128*2*2)
        x = self.encoderFC(x)
        # print(f"Shape after FC encoding: {x.shape}")
        x = self.decoderFC(x)
        # print(f"Shape after FC decoding: {x.shape}")
        x = x.view(-1, 128, 2, 2)
        # print(f"Shape for decoder CNN: {x.shape}")
        x = self.decoderCNN(x)
        # print(f"Shape after decoding: {x.shape}")
        return x


def train(X, epochs):
    model = AutoEncoder().to(DEVICE)
    summary(model, (1, 128, 128))
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-1, weight_decay=1e-8)
    losses = []
    outputs = []
    model.train()

    for i in range(epochs):
        print(f"Epoch: {i}")
        first = True
        local_loss = []
        for img, _ in X:
            if first:
                print(torch.cuda.memory_allocated()/1024**2)
            img = img.to(DEVICE)
            if first:
                print(torch.cuda.memory_allocated()/1024**2)
            out = model(img)
            loss = loss_function(img, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            local_loss.append(loss.item())
            if first:
                outputs.append((img, out))
            first = False
            del loss, out, img
        losses.append(np.mean(local_loss))
        torch.cuda.empty_cache()


dataset = data.Characters(data.PATH_TO_DATA_SHORT, 128, 128, 1)
X_train, _, _ = data.split(dataset)

train(X_train, 10)
