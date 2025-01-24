import pathlib
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

random.seed(10)
import os
import tempfile
import time

import ray
import ray.cloudpickle as pickle
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import f1_score
from src.CNN_training import load_data, test_performance
from src.Unet_1D_model import build_unet
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_loop_tune(config):
    """
    Trains a U-Net model using the provided configuration. Used for raytune optimization.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model and training process.
            Expected keys:
                - 'num_classes' (int): Number of output classes.
                - 'layer_n' (int): Number of layers in the U-Net model.
                - 'dropout' (float): Dropout rate.
                - 'loss_fn' (str): Loss function to use ('BCE', 'DL', 'DLLog').
                - 'lr' (float): Learning rate.
                - 'batch_size' (int): Batch size.
                - 'n_epochs' (int): Number of epochs.
                - 'saveString' (str): Prefix for saving model checkpoints.

    Returns:
        None
    """

    DATAPATH = pathlib.Path("data/labeled_data_fileNames.txt")

    FILEPATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    train_loader, valid_loader, test_loader = load_data(
        FILEPATH / DATAPATH,
        window_size=2000,
        stride=1000,
        data_augmentation=False,
        n_features=1,
        batch_size=config["batch_size"],
    )
    # train_loader, valid_loader = data_loaders
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
        if "pos_weight" in config.keys():
            pos_weight = torch.tensor([config["pos_weight"]])
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
    elif config["loss_fn"] == "DL":
        loss_fn = smp.losses.DiceLoss(mode="binary")
    elif config["loss_fn"] == "DLLog":
        loss_fn = smp.losses.DiceLoss(mode="binary", log_loss=True)
    else:
        print("Undefined loss function")

    # Build model, initial weight and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=1e-5
    )  # Using Adam optimizer
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=8, factor=0.5, min_lr=1e-7
    )  # Using ReduceLROnPlateau schedule

    for epoch in range(0, 20):
        start_time = time.time()
        model.train()
        avg_loss = 0.0
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
            ] = pred.reshape((-1))
            train_targets[
                i * batch_size * train_shape[2] : (i + 1) * batch_size * train_shape[2]
            ] = (y_batch.detach().cpu().argmax(axis=1).reshape((-1)))
            del y_pred, loss, x_batch, y_batch, pred

        model.eval()
        # Validation loss
        total = 0
        correct = 0
        avg_val_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            with torch.no_grad():
                y_pred = model(x_batch).detach()

                avg_val_loss += loss_fn(y_pred.cpu(), y_batch.float()).item() / len(
                    valid_loader
                )
                pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
                true_label = y_batch.detach().cpu().argmax(axis=1)
                total += true_label.size(0)
                correct += (pred == true_label).sum().item()

                val_preds[
                    i * batch_size * val_shape[2] : (i + 1) * batch_size * val_shape[2]
                ] = pred.reshape((-1))
                val_targets[
                    i * batch_size * val_shape[2] : (i + 1) * batch_size * val_shape[2]
                ] = true_label.reshape((-1))
                del y_pred, x_batch, y_batch, pred

        train_score = f1_score(train_targets, train_preds, average="macro")
        val_score = f1_score(val_targets, val_preds, average="macro")

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = pathlib.Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {
                    "loss": avg_val_loss,
                    "accuracy": correct / total,
                    "train_score": train_score,
                    "val_score": val_score,
                },
                checkpoint=checkpoint,
            )

        elapsed_time = time.time() - start_time
        scheduler.step(avg_val_loss)

    # print("VALIDATION_SCORE (F1): ", f1_score(val_target_onehot.argmax(axis=1).flatten(),val_preds ,average = 'macro'))
    print("Finished Training")


if __name__ == "__main__":

    window_size = 2000
    stride = window_size // 2
    # batch_size=32
    n_features = 1
    num_samples = 20
    max_num_epochs = 50

    ray.init(num_cpus=8)

    config = {
        "dropout": tune.uniform(0.0, 0.5),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "lr": tune.loguniform(1e-6, 1e-2),
        "loss_fn": "BCE",
        "layer_n": tune.randint(8, 48),
        "num_classes": 2,
        "pos_weight": 5,
    }

    scheduler = ASHAScheduler(
        metric="val_score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2,
    )

    from ray.tune.search.optuna import OptunaSearch

    algo = OptunaSearch(metric="val_score", mode="max")
    trainable_with_resources = tune.with_resources(training_loop_tune, {"cpu": 4})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=num_samples,
            scheduler=scheduler,
            # resources_per_trial={"cpu": 2, "gpu": 0},
        ),
        param_space=config,
    )

    result = tuner.fit()

    # result = tune.run(
    #     training_loop_tune,
    #     resources_per_trial={"cpu": 2, "gpu": 0},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    # )

    best_trial = result.get_best_result("val_score", "max", "last")
    print(best_trial)
    print(best_trial.metrics)
    print(f"Best trial config: {best_trial.config}")
    # print(f"Best trial final validation loss: {best_trial['loss']}")
    # print(f"Best trial final validation accuracy: {best_trial['accuracy']}")
    # print(f"Best trial final validation f1_score: {best_trial['val_score']}")

    # build_unet(input_channels=1, num_classes=config['num_classes'],layer_n=config['layer_n'],conv_kernel_size=3,scaling_kernel_size=2,dropout=config['dropout'])
    best_trained_model = build_unet(
        input_channels=1,
        num_classes=best_trial.config["num_classes"],
        layer_n=best_trial.config["layer_n"],
        conv_kernel_size=3,
        scaling_kernel_size=2,
        dropout=best_trial.config["dropout"],
    )
    device = "cpu"
    best_trained_model.to(device)

    best_checkpoint = best_trial.get_best_checkpoint(metric="val_score", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = pathlib.Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dict"])
        test_acc = test_performance(best_trained_model, device=device)
        # print("Best trial test set accuracy: {}".format(test_acc))
