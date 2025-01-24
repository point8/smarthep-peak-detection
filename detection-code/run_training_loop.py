import pathlib

import torch
from src.CNN_training import load_data, test_performance, training_loop

if __name__ == "__main__":
    DATAPATH = pathlib.Path("data/labeled_data_fileNames.txt")
    window_size = 2000
    stride = window_size // 2
    n_features = 1

    config = {
        "dropout": 0.2,
        "batch_size": 16,
        "lr": 0.0003,
        "loss_fn": "BCE",
        "layer_n": 24,
        "num_classes": 2,
        "weights": torch.tensor([5.0]),
        "n_epochs": 1,
        "save_best_training_model": True,
        "saveString": "generated_data_RobustScaled3070_1DUnet",
    }

    train_loader, valid_loader, test_loader = load_data(
        DATAPATH,
        window_size=window_size,
        stride=stride,
        data_augmentation=False,
        n_features=1,
        batch_size=config["batch_size"],
    )

    model, loss_values, version_name = training_loop(
        (train_loader, valid_loader), config
    )

    test_performance(test_loader, model)
