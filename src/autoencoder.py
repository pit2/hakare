import torch


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(128*127, 4064),
            torch.nn.ReLU(),
            torch.nn.Linear(4064, 2032),
            torch.nn.ReLU(),
            torch.nn.Linear(2032, 1016),
            torch.nn.ReLU(),
            torch.nn.Linear(1016, 508),
            torch.nn.ReLU(),
            torch.nn.Linear(508, 254)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(254, 508),
            torch.nn.ReLU(),
            torch.nn.Linear(508, 1016),
            torch.nn.ReLU(),
            torch.nn.Linear(1016, 2032),
            torch.nn.ReLU(),
            torch.nn.Linear(2032, 4064),
            torch.nn.ReLU(),
            torch.nn.Linear(4064, 128*127),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(X, epochs):
    model = AutoEncoder()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-1, weight_decay=1e-8)
    losses = []
    outputs = []

    for _ in range(epochs):
        for img in X:
            out = model(img)
            loss = loss_function(img, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            outputs.append((img, out))


X = torch.rand(128*127)


train([X], 1)
