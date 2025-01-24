import pathlib
import random

import numpy as np

# import sklearn as sk
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

random.seed(10)
import time

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

from torch.utils.data import ConcatDataset, Dataset

from .data_augmentation import augment_data
from .Unet_1D_model import build_unet


def training_loop(data_loaders, config):
    """
    Trains a U-Net model using the provided data loaders and configuration.
    Args:
        data_loaders (tuple): A tuple containing the training and validation data loaders.
        config (dict): A dictionary containing the configuration parameters for the model and training process.
            Expected keys:
                - 'num_classes' (int): Number of output classes.
                - 'layer_n' (int): Number of layers in the U-Net model.
                - 'dropout' (float): Dropout rate.
                - 'loss_fn' (str): Loss function to use ('CE','BCE', 'DL', 'DLLog').
                - 'lr' (float): Learning rate.
                - 'batch_size' (int): Batch size.
                - 'n_epochs' (int): Number of epochs.
                - 'saveString' (str): Prefix for saving model checkpoints.
    Returns:
        model (torch.nn.Module): The trained U-Net model.
        loss_values (np.ndarray): Array containing loss and metric values for each epoch.
        version_name (str): The version name used for saving model checkpoints.
    """
    required_params = [
        "num_classes",
        "layer_n",
        "dropout",
        "loss_fn",
        "lr",
        "batch_size",
        "n_epochs",
        "saveString",
    ]
    if not all(key in config for key in required_params):
        missing_params = [key for key in required_params if key not in config]
        raise ValueError(f"Missing required parameters in config: {missing_params}")

    # train_loader, valid_loader, test_loader  = load_data(DATAPATH,window_size=2000,stride=1000,data_augmentation=False,n_features=1,batch_size=config['batch_size'])
    train_loader, valid_loader = data_loaders
    batch_size = config["batch_size"]
    model = build_unet(
        input_channels=1,
        num_classes=config["num_classes"],
        layer_n=config["layer_n"],
        conv_kernel_size=3,
        scaling_kernel_size=2,
        dropout=config["dropout"],
    )

    train_shape = (len(train_loader.dataset),) + train_loader.dataset[0][0].shape
    val_shape = (len(valid_loader.dataset),) + valid_loader.dataset[0][0].shape

    ## Build tensor data for torch
    train_preds = np.zeros((int(train_shape[0] * train_shape[2])))
    val_preds = np.zeros((int(val_shape[0] * val_shape[2])))
    train_targets = np.zeros((int(train_shape[0] * train_shape[2])))
    val_targets = np.zeros((int(val_shape[0] * val_shape[2])))

    ##Loss function

    if config["loss_fn"] == "BCE":
        if "weights" in config.keys():
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=config["weights"])
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
    elif config["loss_fn"] == "DL":
        if config["num_classes"] == 2:
            loss_fn = smp.losses.DiceLoss(mode="binary")
        else:
            loss_fn = smp.losses.DiceLoss(mode="multilabel")
    elif config["loss_fn"] == "DLLog":
        if config["num_classes"] == 2:
            loss_fn = smp.losses.DiceLoss(mode="binary", log_loss=True)
        else:
            loss_fn = smp.losses.DiceLoss(mode="multilabel", log_loss=True)
    elif config["loss_fn"] == "CE" and config["num_classes"] > 2:
        if config["weights"]:
            loss_fn = torch.nn.CrossEntropyLoss(weight=config["weights"])
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
    else:
        print("Undefined loss function")

    # Build model, initial weight and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=1e-5
    )  # Using Adam optimizer
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=8, factor=0.5, min_lr=1e-7
    )  # Using ReduceLROnPlateau schedule
    temp_val_loss = 9999999999
    temp_train_loss = 9999999999

    loss_values = []
    version_name = (
        config["saveString"]
        + f"_layer{config['layer_n']}_dropout{config['dropout']}_loss{config['loss_fn']}_lr{config['lr']}_bs{config['batch_size']}_epochs{config['n_epochs']}"
    )
    for epoch in range(0, config["n_epochs"]):
        start_time = time.time()
        model.train()
        avg_loss = 0.0
        running_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(x_batch)

            loss = loss_fn(y_pred.cpu(), y_batch.float())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

            pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)

            train_preds[
                i * batch_size * train_shape[2] : (i + 1) * batch_size * train_shape[2]
            ] = pred.flatten()
            train_targets[
                i * batch_size * train_shape[2] : (i + 1) * batch_size * train_shape[2]
            ] = y_batch.argmax(axis=1).flatten()
            del y_pred, loss, x_batch, y_batch, pred

        if "save_best_training_model" in config.keys():
            if config["save_best_training_model"]:
                if avg_loss < temp_train_loss:
                    temp_train_loss = avg_loss
                    model_scripted = torch.jit.script(model)  # Export to TorchScript
                    model_scripted.save(
                        "Models/" + version_name + "_trainingcheckpoint_model.pt"
                    )  # Save

        model.eval()

        avg_val_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()

            avg_val_loss += loss_fn(y_pred.cpu(), y_batch.float()).item() / len(
                valid_loader
            )
            pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
            val_preds[
                i * batch_size * val_shape[2] : (i + 1) * batch_size * val_shape[2]
            ] = pred.flatten()
            val_targets[
                i * batch_size * val_shape[2] : (i + 1) * batch_size * val_shape[2]
            ] = y_batch.argmax(axis=1).flatten()
            del y_pred, x_batch, y_batch, pred

        if avg_val_loss < temp_val_loss:
            print("checkpoint_save")
            temp_val_loss = avg_val_loss
            model_scripted = torch.jit.script(model)  # Export to TorchScript
            model_scripted.save(
                "Models/" + version_name + "_validationcheckpoint_model.pt"
            )  # Save
            # torch.save(model.state_dict(), "checkpoints/"+version_name+'_checkpointvalidation.pt')

        str_training_preds = "Training Predictions: " + "".join(
            f" {i}: {sum(train_preds==i)}" for i in range(config["num_classes"])
        )
        str_training_targets = "Training Targets: " + "".join(
            f" {i}: {sum(train_targets==i)}" for i in range(config["num_classes"])
        )
        print(str_training_preds)
        print(str_training_targets)

        train_score = f1_score(train_targets, train_preds, average="macro")
        val_score = f1_score(val_targets, val_preds, average="macro")
        train_score_precision = precision_score(
            train_targets, train_preds, average="macro"
        )
        val_score_precision = precision_score(val_targets, val_preds, average="macro")
        train_score_recall = recall_score(train_targets, train_preds, average="macro")
        val_score_recall = recall_score(val_targets, val_preds, average="macro")

        elapsed_time = time.time() - start_time
        scheduler.step(avg_val_loss)

        print(
            "Epoch {}/{} \t loss={:.4f} \t train_f1={:.4f} \t train_precision={:.4f} \t train_recall={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} \t val_precision={:.4f} \t val_recall={:.4f} \t time={:.2f}s".format(
                epoch + 1,
                config["n_epochs"],
                avg_loss,
                train_score,
                train_score_precision,
                train_score_recall,
                avg_val_loss,
                val_score,
                val_score_precision,
                val_score_recall,
                elapsed_time,
            )
        )

        loss_values.append(
            [
                epoch + 1,
                avg_loss,
                train_score,
                train_score_precision,
                train_score_recall,
                avg_val_loss,
                val_score,
                val_score_precision,
                val_score_recall,
            ]
        )

    # print("VALIDATION_SCORE (F1): ", f1_score(val_target_onehot.argmax(axis=1).flatten(),val_preds ,average = 'macro'))
    return model, np.array(loss_values), version_name


def load_data(
    list_of_files_path,
    window_size=2000,
    stride=1000,
    data_augmentation=False,
    n_features=1,
    num_classes=2,
    batch_size=32,
):
    """
    Load and preprocess data for training, validation, and testing.
    Parameters:
    -----------
    DATAPATH : pathlib.Path
        Path to the directory containing the data files.
    labelType : str, optional
        Type of labeling used in the data files. Default is 'wide'.
    window_size : int, optional
        Size of the window for creating data windows. Default is 2000.
    stride : int, optional
        Stride for creating data windows. Default is 1000.
    data_augmentation : bool, optional
        Whether to apply data augmentation. Default is False.
    n_features : int, optional
        Number of features in the data. Default is 1.
    Returns:
    --------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    valid_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    """

    data_dir = list_of_files_path.parent
    with open(list_of_files_path) as f:
        list_of_files = [x.strip() for x in f.readlines()]

    all_data = []
    all_labels = []

    for n, file in enumerate(list_of_files):
        data, labels = np.loadtxt(
            pathlib.Path(data_dir / file).resolve(), delimiter=","
        )

        rsc = RobustScaler(quantile_range=(30.0, 70.0), unit_variance=True)
        data = data.reshape(-1, 1)

        data = rsc.fit_transform(data)

        all_data.append(data)
        all_labels.append(labels)
    X = np.array(all_data)
    y = np.array(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.22, random_state=101
    )  # 0.22 x 0.9 = 0.2

    def create_windows(data, window_size, stride):
        missing_vals = window_size - (len(data) % window_size)
        data_looped = np.append(data, data[: missing_vals + window_size])
        return np.array(
            [data_looped[i : i + window_size] for i in range(0, len(data) + 1, stride)]
        )

    def windowing(data, labels):
        data_windows, label_windows = [], []
        for data, label in zip(data, labels):
            data_windows.append(create_windows(data, window_size, stride))
            label_windows.append(create_windows(label, window_size, stride))
        return np.concatenate(data_windows, dtype=np.float32), np.concatenate(
            label_windows
        )

    X_train_windows, y_train_windows = windowing(X_train, y_train)
    X_val_windows, y_val_windows = windowing(X_val, y_val)
    X_test_windows, y_test_windows = windowing(X_test, y_test)

    X_train_input = X_train_windows.reshape(
        (len(X_train_windows), n_features, window_size)
    )
    X_val_input = X_val_windows.reshape((len(X_val_windows), n_features, window_size))
    X_test_input = X_test_windows.reshape(
        (len(X_test_windows), n_features, window_size)
    )

    train_target_onehot = torch.nn.functional.one_hot(
        torch.tensor(y_train_windows).long(), num_classes=num_classes
    ).transpose(1, 2)
    val_target_onehot = torch.nn.functional.one_hot(
        torch.tensor(y_val_windows).long(), num_classes=num_classes
    ).transpose(1, 2)
    test_Y_windows_onehot = torch.nn.functional.one_hot(
        torch.tensor(y_test_windows).long(), num_classes=num_classes
    ).transpose(1, 2)

    train_dataset = TimeSeriesDataset(X_train_input, train_target_onehot, augment=False)
    if data_augmentation:
        train_dataset = ConcatDataset(
            [
                train_dataset,
                TimeSeriesDataset(X_train_input, train_target_onehot, augment=True),
            ]
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        TimeSeriesDataset(X_val_input, val_target_onehot, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        TimeSeriesDataset(X_test_input, test_Y_windows_onehot, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, augment=False):
        if augment:
            data = augment_data(data)
        self.data = torch.from_numpy(data.astype("float32").copy()).clone().to(device)
        self.targets = targets.clone().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        return x, y


def test_performance(test_loader, model):

    test_shape = (len(test_loader.dataset),) + test_loader.dataset[0][0].shape

    model.eval()
    test_preds = np.zeros((int(test_shape[0] * test_shape[2])))
    test_targets = np.zeros((int(test_shape[0] * test_shape[2])))

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            batch_size = len(x_batch)

            y_pred = model(x_batch).detach()

            pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
            y_batch_pred = y_batch.detach().cpu().numpy().argmax(axis=1)

            test_preds[
                i * batch_size * test_shape[2] : (i + 1) * batch_size * test_shape[2]
            ] = pred.flatten()
            test_targets[
                i * batch_size * test_shape[2] : (i + 1) * batch_size * test_shape[2]
            ] = y_batch_pred.flatten()

            del y_pred, x_batch, y_batch, pred
    print(test_targets.shape)
    print(max(test_targets))
    for i in range(int(max(test_targets) + 1)):
        print(f"Test Predictions: {i}: {sum(test_preds==i)}")
        print(f"Test Targets: {i}: {sum(test_targets==i)}")

    print("TEST_SCORE (F1): ", f1_score(test_targets, test_preds, average="macro"))
    print(
        "TEST_SCORE (Precision): ",
        precision_score(test_targets, test_preds, average="macro"),
    )
    print(
        "TEST_SCORE (Recall): ", recall_score(test_targets, test_preds, average="macro")
    )

    return 0
