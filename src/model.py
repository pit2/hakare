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
import coremltools as ct


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUT_PATH = os.path.join("data", "models")


class Recognizer(torch.nn.Module):
    """CNN to recognize 90x90 greyscale images of handwritten Japanese characters.

    Parameters:
    trial (Optuna trial): Pass trial to conduct hyper-parameter optimisation study
    config (1,2,3): Build CNN from pre-defined configuration:
                    1 - winner from study 1
                    2 - winner from study 2
                    3 - winner from study 1 without dropout/batch-normalisation
    """

    def __init__(self, trial=None, config=1):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = 0
        self.test_loss = np.inf
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.min_valid_loss = np.inf
        self.epochs = 0

        def conv_dim(dim, padding=0, kernel_size=2, stride=2):
            return math.floor(((dim + 2 * padding - (kernel_size - 1) - 1) / stride) + 1)

        if trial is None:
            if config == 1:  # study 1 winner
                dim = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(
                    90, 1, 5, 2), 0, 2, 2), 1, 3, 2), 0, 2, 2), 1, 3, 1), 1, 3, 1), 1, 3, 1)
                self.FC_DIM = 256 * dim * dim
                dr = 0.3262790132739898
                print("Construct config 1")
                self.cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(dr),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Conv2d(16, 32, kernel_size=3,
                                    stride=2, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(dr),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Conv2d(32, 64, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(dr),
                    torch.nn.Conv2d(64, 128, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(dr),
                    torch.nn.Conv2d(128, 256, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout2d(dr)
                )
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(self.FC_DIM, 3036),
                    torch.nn.Dropout(dr/2)
                )
            elif config == 2:  # study 2 winner
                dim = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(
                    90, 1, 5, 2), 0, 2, 2), 1, 5, 2), 0, 2, 2), 1, 3, 1), 1, 3, 1), 1, 3, 1)
                print("Construct config 2")
                self.FC_DIM = 256 * dim * dim
                self.cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Conv2d(16, 32, kernel_size=5,
                                    stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Conv2d(32, 64, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 256, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.ReLU(),
                )
                self.fc = torch.nn.Linear(self.FC_DIM, 3036)
            elif config == 3:  # study 1 winner without regularisation
                dim = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(
                    90, 1, 5, 2), 0, 2, 2), 1, 3, 2), 0, 2, 2), 1, 3, 1), 1, 3, 1), 1, 3, 1)
                self.FC_DIM = 256 * dim * dim
                dr = 0.3262790132739898
                print("Construct config 3")
                self.cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Conv2d(16, 32, kernel_size=3,
                                    stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Conv2d(32, 64, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 256, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.ReLU()
                )
                self.fc = torch.nn.Linear(self.FC_DIM, 3036)

        else:
            print("construct trial configuration")
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
            # dropout_rate = trial.suggest_loguniform("dropout rate", 0.3, 0.7)
            kernel_size_1 = trial.suggest_categorical(
                "kernel size of 1st conv", [3, 5, 7])
            kernel_size_2 = trial.suggest_categorical(
                "kernel size of 2nd conv", [3, 5])
            padding = math.floor(num_conv / 2) - 1
            stride = 2 if num_conv <= 6 else 3
            layers = []
            kernels = [kernel_size_1, kernel_size_2] + \
                [3 for n in range(num_conv - 2)]
            strides = [stride, stride] + [1 for n in range(num_conv - 2)]
            in_channels = 1

            for i in range(num_conv):
                new_dim = conv_dim(dim, padding, kernels[i], strides[i])
                if new_dim >= MIN_DIM:
                    layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernels[i],
                                  stride=strides[i], padding=padding))
                    if i in (0, 3, 6):
                        layers.append(torch.nn.BatchNorm2d(out_channels))
                    layers.append(torch.nn.ReLU())
                    # layers.append(torch.nn.Dropout2d(dropout_rate))
                    dim = new_dim
                    new_dim = conv_dim(dim)

                    # add additional pooling if final dim is still too big
                    if (num_pool >= 1 and new_dim >= MIN_DIM) or (num_pool < 1 and dim >= 12):
                        layers.append(torch.nn.MaxPool2d(
                            kernel_size=2, stride=2, padding=0))
                        dim = new_dim
                        num_pool -= 1
                else:
                    break
                in_channels = out_channels
                out_channels = in_channels * 2
            self.FC_DIM = dim * dim * in_channels
            self.cnn = torch.nn.Sequential(*layers)
            self.fc = torch.nn.Linear(self.FC_DIM, 3036)
            #  torch.nn.Dropout(dropout_rate / 2))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, self.FC_DIM)
        return self.fc(x)

    def convert_to_onnx(self, filename_onnx, sample_input):
        input_names = ["in_image"]
        output_names = ["out_class"]

        torch.onnx.export(self, torch.from_numpy(np.array(sample_input)), filename_onnx,
                          input_names=input_names, output_names=output_names)


def train(model, train_gen, valid_gen, params={"lr": 1e-3, "weight_decay": 1e-4,
                                               "betas": (0.5, 0.5)},
          stats={"mean": 0, "std": 1}, epochs=10, report=True):
    """Using Adam optimizer, train and validate the model. Returns the model with the smallest
    loss on the validation set after given number of epochs.

    Parameters:
    model (Recognizer): CNN model to be trained and validated
    train_gen (DataLoader): training data (Characters class)
    valid_gen (DataLoader): validation data (Characters class)
    params (dictionary): with keys lr, weight_decay, betas to hold parameters for
                        adams optimizer
    stats (dictionary): with keys mean and std to hold statistical params of training set
    epochs (int): number of epochs for training
    report (Boolean): true to print test/validation loss/acc after each epoch and plot loss/acc
                      after every 10th epoch

    Returns best Recognizer model (i.e. min validation error)
    """

    model.to(DEVICE)

    best_model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                 weight_decay=params["weight_decay"], betas=params["betas"])

    for i in range(model.epochs, model.epochs + epochs):

        model, train_loss, train_acc = _step(
            model, optimizer, train_gen, params, stats)
        model.train_losses.append(train_loss)
        model.train_acc.append(train_acc*100)

        valid_loss, valid_acc = evaluate(model, valid_gen, stats)
        model.valid_losses.append(valid_loss)
        model.valid_acc.append(valid_acc*100)

        if report:
            print(
                f"Epoch {i} --- train loss: {train_loss} --- validation loss: {valid_loss}")
            print(f"Accuracy on validation set: {valid_acc*100}%")
            if i > 0 and i % 10 == 0:
                x_axis = [x for x in range(model.epochs + i + 1)]
                plt.plot(x_axis, model.train_losses, label="train loss")
                plt.plot(x_axis, model.valid_losses, label="valid loss")
                plt.legend()
                plt.show()
                x_axis = [x for x in range(model.epochs + i + 1)]
                plt.plot(x_axis, model.train_acc, label="train accuracy %")
                plt.plot(x_axis, model.valid_acc, label="valid accuracy %")
                plt.legend()
                plt.show()

        if model.min_valid_loss > valid_loss:
            best_model = model
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

            progress.set_postfix(loss=np.mean(losses), accuracy=np.mean(
                accuracies) * 100, refresh=False)
            progress.update()

    return model, np.mean(losses), np.mean(accuracies)


def evaluate(model, test_gen, stats):
    """Evaluate the model and return the loss.

    Parameters:
        model (Recognizer): Model to be evaluated.
        test_gen (DataLoader): Test (or validation) data to evaluate the model.
        stats (Dictionary): holding mean and std of training set.

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


def fit(model_path=None, epochs=10, config=2):
    """Fit a model to training data. Print loss/accuracy on test set after training is complete.

    Parameters:
    model_path (str): Path to *.pt model to be loaded; None to train fresh model.
    epochs (int): Number of iterations for training.
    config (1,2,3): Topology of the model to be used to fit.

    Returns: Recognizer with least loss on validation set."""
    torch.cuda.empty_cache()
    dataset = data.Characters(data.PATH_TO_DATA, 1, 90, 90)
    print(f"Size of dataset: {len(dataset)}")
    train_data, valid, test = data.split(
        dataset, batch_size=256, train=0.5, valid=0.2, test=0.3, num_workers=0)
    dataset.mean, dataset.std = data.get_mean_std(train_data)
    stats_dict = {"mean": dataset.mean, "std": dataset.std}
    model = load_model(model_path, None, config)
    model = train(model, train_data, valid,
                  params={"lr": 0.00024029296658391487,
                          "weight_decay": 1e-4, "betas": (0.9, 0.999)},
                  epochs=epochs, stats=stats_dict, report=True)
    model.test_loss, test_acc = evaluate(
        model, test, stats=stats_dict)
    model.test_acc = test_acc * 100
    print(f"Loss on test set: {model.test_loss}")
    print(f"Accuracy on test set: {model.test_acc}%")

    return model


def objective(trial, epochs=8, model_path=None):
    """Launch a new trial for Optuna hyperparameters optimisation.

    Parameters:
    trial (Optuna trial): Trial to be used.
    epochs (int): Number of epochs to train each trial.
    model_path (str): Path to *.pt model to resume previous trial.

    Returns: Accuracy of model on test set as objective."""
    torch.cuda.empty_cache()
    model = load_model(model_path, trial)
    dataset = data.Characters(data.PATH_TO_DATA, 1, 90, 90)
    batch_size = trial.suggest_categorical(
        "batch size", [16, 32, 64, 128, 256, 512, 1024])
    train_data, valid, test = data.split(
        dataset, batch_size=batch_size, train=0.3, valid=0.05, test=0.15, num_workers=0)
    dataset.mean, dataset.std = data.get_mean_std(train_data)
    stats_dict = {"mean": dataset.mean, "std": dataset.std}
    beta1 = 0.9
    beta2 = 0.999
    lr = trial.suggest_loguniform("lr", 0.0001, 0.004)
    model = train(model, train_data, valid,
                  params={"lr": lr, "weight_decay": 1e-4,
                          "betas": (beta1, beta2)},
                  epochs=epochs, stats=stats_dict, report=False)
    model.test_loss, test_acc = evaluate(
        model, test, stats=stats_dict)
    model.test_acc = test_acc * 100
    return model.test_acc


def load_model(model_path, trial=None, config=2):
    """Load model or create new model with given specifications.

    Parameters:
        model_path (str): Path to *.pt to load, or None to initialise new model.
        trial (Optuna trial): Set to None to launch specific configuration. Only relevant if
                              model_path is None.
        config (1,2,3): Configuration of the model to be used. Only relevant if path and trial
                        are None.

    Returns: Recognizer
    """

    if model_path is None:
        model = Recognizer(trial, config).to(DEVICE)
    else:
        model = torch.load(model_path, map_location=torch.device(DEVICE))

    return model


# print_topology(Recognizer())
def optimize_hyper(name, save_study=False, load_study=False, n_trials=10, timeout=3600):
    """Create a new Optuna hyper-parameters optimisation study.

    Parameters:
        name (str): (Unique) Name of the study.
        save_study (bool): True to save study as sqllite db.
        load_study (bool): True to load study of given name, provided it exists.
        n_trials (int): Number of trials in this study.
        timeout (int): Maximum number of seconds the study may run.
    """
    storage_name = f"sqlite:///{name}.db" if save_study else None

    if load_study:
        study = optuna.load_study(study_name=name, storage=storage_name)
    else:
        study = optuna.create_study(study_name=name, storage=storage_name, direction="maximize",
                                    sampler=optuna.samplers.NSGAIISampler(population_size=20,
                                                                          mutation_prob=None,
                                                                          crossover_prob=0.9,
                                                                          swapping_prob=0.5))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)


def trace_model(model_path):
    model = load_model(model_path)
    model.eval()
    dataset = data.Characters(data.PATH_TO_DATA, 1, 90, 90)
    train_data, valid, test = data.split(
        dataset, batch_size=1, train=0.9, valid=0.05, test=0.05, num_workers=0)
    dataset.mean, dataset.std = data.get_mean_std(train_data)
    stats_dict = {"mean": dataset.mean, "std": dataset.std}
    for img, target in train_data:
        sample_input, _ = data.transform(
            img, target, stats_dict["mean"], stats_dict["std"], 1, 90, 90)
        break
    traced_model = torch.jit.script(model)  # (model, sample_input)
    print(type(traced_model))
    return traced_model(sample_input), sample_input


def convert_to_coreml(traced_model, sample_input):
    print(type(traced_model))
    model = ct.convert(traced_model, inputs=[
                       ct.TensorType(shape=sample_input.shape)], source="pytorch")
    model.save(os.path.join("data", "models", "ios_model.mlmodel"))


traced_model, sample_input = trace_model(
    os.path.join("data", "models", "model-trial-2-12-ep40.pt"))
convert_to_coreml(traced_model, sample_input)
