import torch
import data
import numpy as np
from torchsummary import summary
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import tqdm
import optuna
import math


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Recognizer(torch.nn.Module):
    """CNN to recognize 128x128 greyscale images of handwritten Japanese characters."""

    def __init__(self, trial=None):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracy = []
        self.test_accuracy = 0
        self.test_loss = np.inf
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.min_valid_loss = np.inf
        self.epochs = 0

        if trial is None:
            self.FC_DIM = 256 * 8 * 8
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=3),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(0.4),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=3),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(0.4),
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(0.4),
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=3),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Dropout2d(0.4)
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.FC_DIM, 3036),
                torch.nn.Dropout(0.2)
            )
        else:
            def conv_dim(dim, padding=0, kernel_size=2, stride=2):
                return math.floor(((dim + 2 * padding - (kernel_size - 1) - 1) / stride) + 1)

            dim = 128
            MIN_DIM = 4
            num_conv = 7 # trial.suggest_int("num_conv_layers", 2, 8)
            num_pool = math.floor(num_conv / 2)
            out_channels = (int) (trial.suggest_categorical("output_channels_of_first_conv", [8, 16, 32]))
            dropout_rate = trial.suggest_loguniform("dropout rate", 0.1, 0.7)
            kernel_size_1 = (int) (trial.suggest_categorical("kernel size of first conv", [3, 5, 7]))
            kernel_size_2 = (int) (trial.suggest_categorical("kernel size of second conv", [n for n in [3,5,7] if n <= kernel_size_1 ]))
            padding = math.floor(num_conv / 2) - 1
            stride = 1 if num_conv > 4 else 2
            layers = []
            kernels = [kernel_size_1, kernel_size_2] + [3 for n in range(num_conv - 2)]
            in_channels = 1
            print(f"Kernels: {kernels}")
            print(f"Pools: {num_pool}")
            print(f"Out channels: {out_channels}")
            print(f"Padding: {padding}")
            print(f"Stride: {stride}")

            for i in range(num_conv):
                new_dim = conv_dim(dim, padding, kernels[i], stride)
                if new_dim >= MIN_DIM:
                    layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernels[i], stride=stride, padding=padding))
                    layers.append(torch.nn.BatchNorm2d(out_channels))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout2d(dropout_rate))
                    dim = new_dim
                    new_dim = conv_dim(dim)

                    # add additional pooling if final dim is still too big
                    if (num_pool >= 1 and new_dim >= MIN_DIM) or (num_pool == 0 and dim >= 16):
                        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                        dim = new_dim
                        num_pool -= 1
                else:
                    break
                in_channels = out_channels
                out_channels = in_channels * 2
            self.FC_DIM = dim * dim * in_channels
            print(f"Final dim: {dim}x{dim}")
            print(f"FC_DIM: {self.FC_DIM}")
            self.cnn = torch.nn.Sequential(*layers)
            print(self.cnn)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.FC_DIM, 3036),
                torch.nn.Dropout(dropout_rate / 2))

    def forward(self, x):
        x = self.cnn(x)
        print(x.shape)
        x = x.view(-1, self.FC_DIM)
        print(x.shape)
        return self.fc(x)


def train(model, train_gen, valid_gen, params={"lr": 1e-3, "weight_decay": 1e-4, "betas": (0.5, 0.5)},
          stats={"mean": 0, "std": 1}, epochs=10, report=True):
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

    for i in range(model.epochs, model.epochs + epochs):
        model, train_loss = _step(model, train_gen, params, stats)
        model.train_losses.append(train_loss)

        valid_loss, accuracy = evaluate(model, valid_gen, stats)
        model.valid_losses.append(valid_loss)
        model.valid_accuracy.append(accuracy)

        if report:
            print(
                f"Epoch {i} --- test error: {train_loss} --- validation error: {valid_loss}")
            print(f"Accuracy on validation set: {accuracy}")
            if i > 0 and i % 10 == 0:
                x_axis = [x for x in range(model.epochs + i + 1)]
                print(x_axis)
                plt.plot(x_axis, model.train_losses, label="train loss")
                plt.plot(x_axis, model.valid_losses, label="valid loss")
                plt.legend()
                plt.show()

        if model.min_valid_loss > valid_loss:
            best_model = model

    return best_model


def _step(model, train_gen, params, stats):
    """Execute one training step (one batch) and return updated model and mean loss."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], betas=params["betas"])
    losses = []
    for img, target in train_gen:
        img, target = data.transform(
            img, target, stats["mean"], stats["std"], 1, 128, 128)
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

    return model, np.mean(losses)


def evaluate(model, test_gen, stats):
    """Evaluate the model and return the loss.

    Parameters:
        model (Recognizer): Model to be evaluated.
        test_gen (DataLoader): Test (or validation) data to evaluate the model.

    Returns
        mean loss on test/validation data.
        mean accuracy as percentage of correct predictions.
    """
    model.eval()
    losses = []
    accuracies = []
    accuracy = Accuracy().to(DEVICE)
    with torch.no_grad():
        for img, target in test_gen:
            img, target = data.transform(
                img, target, stats["mean"], stats["std"], 1, 128, 128)
            img = img.to(DEVICE)
            target = target.to(DEVICE)
            predict = model(img)
            loss = model.loss_fn(predict, target)
            losses.append(loss.item())
            accuracies.append(accuracy(predict, target).cpu().item())

    return np.mean(losses), np.mean(accuracies)


def print_topology(model): summary(model, (1, 128, 128))


def execute():
    torch.cuda.empty_cache()
    dataset = data.Characters(data.PATH_TO_DATA_MINI, 128, 128, 1)
    print(f"Size of dataset: {len(dataset)}")
    train_data, valid, test = data.split(dataset, batch_size=5)
    dataset.mean, dataset.std = data.get_mean_std(train_data)
    stats_dict = {"mean": dataset.mean, "std": dataset.std}

    model = train(Recognizer().to(DEVICE), train_data, valid, params={"lr": 0.2, "weight_decay": 1e-4},
                  epochs=1, stats=stats_dict, report=True)
    model.test_loss, model.test_accuracy = evaluate(
        model, test, stats=stats_dict)
    print(f"Loss on test set: {model.test_loss}")
    print(f"Accuracy on test set: {model.test_accuracy * 100}%")

    return model


def objective(trial):
    model = Recognizer(trial).to(DEVICE)
    print_topology(model)
    dataset = data.Characters(data.PATH_TO_DATA_MINI, 128, 128, 1)
    train_data, valid, test = data.split(dataset, batch_size=256)
    dataset.mean, dataset.std = data.get_mean_std(train_data)
    stats_dict = {"mean": dataset.mean, "std": dataset.std}
    beta1 = trial.suggest_loguniform("beta1", 0.01, 0.9)
    beta2 = trial.suggest_loguniform("beta2", 0.01, 0.9)

    model = train(model, train_data, valid, params={"lr": 0.2, "weight_decay": 1e-4, "betas": (beta1, beta2)},
                  epochs=10, stats=stats_dict, report=True)
    model.test_loss, model.test_accuracy = evaluate(
        model, test, stats=stats_dict)
    return model.test_accuracy


# print_topology(Recognizer())
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.NSGAIISampler(
    population_size=20, mutation_prob=None, crossover_prob=0.9, swapping_prob=0.5))
study.optimize(objective, n_trials=1)
