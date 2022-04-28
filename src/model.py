import torch
import data
import numpy as np
from torchsummary import summary
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
import math
import os.path


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUT_PATH = os.path.join("data", "models")


class Recognizer(torch.nn.Module):
    """CNN to recognize 90x90 greyscale images of handwritten Japanese characters."""

    def __init__(self, trial=None):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracy = []
        self.valid_accuracy = []
        self.test_accuracy = 0
        self.test_loss = np.inf
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.min_valid_loss = np.inf
        self.epochs = 0

        def conv_dim(dim, padding=0, kernel_size=2, stride=2):
            return math.floor(((dim + 2 * padding - (kernel_size - 1) - 1) / stride) + 1)

        if trial is None:
            dim = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(90, 3, 5, 2), 1, 2, 2), 3, 3, 2), 3, 3, 2), 3, 3, 2)
            self.FC_DIM = 256 * dim * dim
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

            dim = 90
            MIN_DIM = 4
            num_conv = trial.suggest_int("num_conv_layers", 4, 8)
            num_pool = math.floor(num_conv / 2)
            if num_conv <= 5:
                out_channels = 16
            elif num_conv <= 6:
                out_channels = 8
            elif num_conv <= 7:
                out_channels = 4
            else:
                out_channels = 2
            dropout_rate = trial.suggest_loguniform("dropout rate", 0.3, 0.7)
            kernel_size_1 = trial.suggest_categorical("kernel size of first conv", [3, 5, 7])
            kernel_size_2 = trial.suggest_categorical("kernel size of second conv",
                                                      [n for n in [3, 5] if n <= kernel_size_1])
            padding = math.floor(num_conv / 2) - 1
            stride = 1 if num_conv < 4 else (2 if num_conv <= 6 else 3)
            layers = []
            kernels = [kernel_size_1, kernel_size_2] + [3 for n in range(num_conv - 2)]
            strides = [stride, stride] + [1 for n in range(num_conv - 2)]
            in_channels = 1
            # print(f"Kernels: {kernels}")
            # print(f"Pools: {num_pool}")
            # print(f"Out channels: {out_channels}")
            # print(f"Padding: {padding}")
            # print(f"Stride: {stride}")

            for i in range(num_conv):
                new_dim = conv_dim(dim, padding, kernels[i], strides[i])
                if new_dim >= MIN_DIM:
                    layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernels[i],
                                  stride=strides[i], padding=padding))
                    layers.append(torch.nn.BatchNorm2d(out_channels))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout2d(dropout_rate))
                    dim = new_dim
                    new_dim = conv_dim(dim)

                    # add additional pooling if final dim is still too big
                    if (num_pool >= 1 and new_dim >= MIN_DIM) or (num_pool < 1 and dim >= 12):
                        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                        dim = new_dim
                        num_pool -= 1
                else:
                    break
                in_channels = out_channels
                out_channels = in_channels * 2
            self.FC_DIM = dim * dim * in_channels
            # print(f"Final dim: {dim}x{dim}")
            # print(f"FC_DIM: {self.FC_DIM}")
            self.cnn = torch.nn.Sequential(*layers)
            # print(self.cnn)
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.FC_DIM, 3036),
                torch.nn.Dropout(dropout_rate / 2))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, self.FC_DIM)
        return self.fc(x)


def train(model, train_gen, valid_gen, params={"lr": 1e-3, "weight_decay": 1e-4,
                                               "betas": (0.5, 0.5)}, stats={"mean": 0, "std": 1},
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

    best_model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                 weight_decay=params["weight_decay"], betas=params["betas"])

    for i in range(model.epochs, model.epochs + epochs):

        model, train_loss, train_acc = _step(model, optimizer, train_gen, params, stats)
        model.train_losses.append(train_loss)
        model.train_accuracy.append(train_acc)

        valid_loss, valid_acc = evaluate(model, valid_gen, stats)
        model.valid_losses.append(valid_loss)
        model.valid_accuracy.append(valid_acc)

        if report:
            print(
                f"Epoch {i} --- test error: {train_loss} --- validation error: {valid_loss}")
            print(f"Accuracy on validation set: {valid_acc*100}%")
            if i > 0 and i % 10 == 0:
                x_axis = [x for x in range(model.epochs + i + 1)]
                plt.plot(x_axis, model.train_losses, label="train loss")
                plt.plot(x_axis, model.valid_losses, label="valid loss")
                plt.legend()
                plt.show()
                x_axis = [x for x in range(model.epochs + i + 1)]
                plt.plot(x_axis, model.train_acc*100, label="train accuracy %")
                plt.plot(x_axis, model.valid_acc*100, label="valid accuracy %")
                plt.legend()
                plt.show()

        if model.min_valid_loss > valid_loss:
            best_model = model
            #  now = datetime.now()
            #  dt_string = now.strftime("%d-%m-%Y%-H:%M:%S")
            #  model_name = "model-"+dt_string+".pt"
            model_name = "model.pt"
            torch.save(best_model, os.path.join(OUT_PATH, model_name))

    return best_model


def _step(model, optimizer, train_gen, params, stats):
    """Execute one training step (one batch) and return updated model and mean loss."""
    losses = []
    accuracies = []
    accuracy = Accuracy().to(DEVICE)
    model.train()
    with tqdm(train_gen, unit='batches') as progress:
        for img, target in train_gen:
            img, target = data.transform(
                img, target, stats["mean"], stats["std"], 1, 90, 90)
            img = img.to(DEVICE)
            target = target.to(DEVICE)

            predict = model(img)
            loss = model.loss_fn(predict, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
            accuracies.append(accuracy(predict, target).cpu().item())
            # del img, target, loss
            # torch.cuda.empty_cache()
            progress.set_postfix(loss=np.mean(losses), accuracy=np.mean(accuracies) * 100, refresh=False)
            progress.update()

    return model, np.mean(losses), np.mean(accuracies)


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
                img, target, stats["mean"], stats["std"], 1, 90, 90)
            img = img.to(DEVICE)
            target = target.to(DEVICE)
            predict = model(img)
            loss = model.loss_fn(predict, target)
            losses.append(loss.item())
            accuracies.append(accuracy(predict, target).cpu().item())

    return np.mean(losses), np.mean(accuracies)


def print_topology(model): summary(model, (1, 90, 90))


def execute(model_path=None):
    torch.cuda.empty_cache()
    dataset = data.Characters(data.PATH_TO_DATA, 1, 90, 90)
    # print(f"Size of dataset: {len(dataset)}")
    train_data, valid, test = data.split(dataset, batch_size=32, train=0.3, valid=0.1, test=0.2, num_workers=2)
    dataset.mean, dataset.std = data.get_mean_std(train_data)
    stats_dict = {"mean": dataset.mean, "std": dataset.std}
    model = load_model(model_path)
    model = train(Recognizer().to(DEVICE), train_data, valid,
                  params={"lr": 0.003, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
                  epochs=50, stats=stats_dict, report=True)
    model.test_loss, model.test_accuracy = evaluate(
        model, test, stats=stats_dict)
    print(f"Loss on test set: {model.test_loss}")
    print(f"Accuracy on test set: {model.test_accuracy * 100}%")

    return model


def objective(trial, epochs=8, model_path=None):
    model = load_model(model_path, trial)
    dataset = data.Characters(data.PATH_TO_DATA, 1, 90, 90)
    batch_size = trial.suggest_categorical("batch size", [16, 32, 64, 128, 256, 512, 1024])
    train_data, valid, test = data.split(dataset, batch_size=batch_size, train=0.4, valid=0.1, test=0.2, num_workers=2)
    dataset.mean, dataset.std = data.get_mean_std(train_data)
    stats_dict = {"mean": dataset.mean, "std": dataset.std}
    beta1 = 0.9 # trial.suggest_loguniform("beta1", 0.01, 0.9)
    beta2 = 0.999 # trial.suggest_loguniform("beta2", 0.01, 0.999)
    lr = trial.suggest_loguniform("lr", 0.005, 0.02)
    model = train(model, train_data, valid,
                  params={"lr": lr, "weight_decay": 1e-4, "betas": (beta1, beta2)},
                  epochs=epochs, stats=stats_dict, report=False)
    model.test_loss, model.test_accuracy = evaluate(
        model, test, stats=stats_dict)
    return model.test_accuracy


def load_model(model_path, trial=None):
    if model_path is None:
        model = Recognizer(trial).to(DEVICE)
    else:
        model = torch.load(model_path)
    return model


# print_topology(Recognizer())
def optimize_hyper(name, save_study=False, load_study=False, n_trials=30):
    storage_name = "sqlite:///{}.db".format(name) if save_study else None
    if load_study:
        study = optuna.load_study(study_name="hyperparameter-optimization-3", storage="sqlite:///{}.db".format(name))
    else:
        study = optuna.create_study(study_name=name, storage=storage_name, direction="maximize",
                                sampler=optuna.samplers.NSGAIISampler(population_size=20,
                                                                      mutation_prob=None,
                                                                      crossover_prob=0.9,
                                                                      swapping_prob=0.5))
    study.optimize(objective, n_trials=n_trials, timeout=60*60)

# model = execute()

optimize_hyper("hyperparameter-optimization-9", save_study=True, load_study=False)
