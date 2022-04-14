import torch
import data
import numpy as np
from torchsummary import summary

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Recognizer(torch.nn.Module):
    """CNN to recognize 128x128 greyscale images of handwritten Japanese characters."""

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
        self.min_valid_loss = np.inf

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 128 * 4 * 4)
        return self.fc(x)


def train(model, train_gen, valid_gen, params={"lr": 1e-1, "weight_decay": 1e-8},
          epochs=10, report=True):
    """Using adams optimizer, train and validate the model. Returns the model with the smallest
    loss on the validation set after given number of epochs.

    Parameters:
    model (Recognizer): CNN model to be trained and validated
    train_gen (DataLoader): training data (Characters class)
    valid_gen (DataLoader): validation data (Characters class)
    params (dictionary): Dictionary with keys lr and weight_decay to hold parameters for
                        adams optimizer
    epochs (int): number of epochs for training
    report (Boolean): true to print test/validation error after each epoch

    Returns best Recognizer model (i.e. min validation error)
    """
    model.to(DEVICE)
    model.train()
    best_model = model

    for i in range(epochs):
        model, train_loss = _step(model, train_gen, params)
        #model.train_losses.append(train_loss)

        valid_loss = evaluate(model, valid_gen)

        if report:
            print(
                f"Epoch {i} --- test error: {train_loss} --- validation error: {valid_loss}")

        #if model.min_valid_loss > valid_loss:
        best_model = model

    return best_model


def _step(model, train_gen, params):
    """Execute one training step (one batch) and return updated model and mean loss."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    losses = []
    i = 0
    for img, target in train_gen:
        img = img.to(DEVICE)
        target = target.to(DEVICE)

        predict = model(img)
        loss = model.loss_fn(predict, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        del img, target, loss
        torch.cuda.empty_cache()
        print(f"Step iteration: {i}")
        i = i + 1

    return model  # , np.mean(losses)


def evaluate(model, test_gen):
    """Evaluates the model and returns the loss.

    Parameters:
    model (Recognizer): Model to be evaluated.
    test_gen (DataLoader): Test (or validation) data to evaluate the model.

    Returns mean loss on test/validation data.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for img, target in test_gen:
            predict = model(img)
            loss = model.loss_fn(predict, target)
            losses.append(loss.item())

    return np.mean(losses)


def print_topology(model): summary(model, (1, 128, 128))


def execute():
    torch.cuda.empty_cache()
    dataset = Characters(PATH_TO_DATA, 100, 128, 128, 1)
    train_data, valid, test = split(dataset, batch_size=8)

    model = train(Recognizer(), train_data, valid)
    model.test_loss = evaluate(model, test)
    print(model.test_loss)


execute()
