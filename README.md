# Industrial time series analysis at Point 8

## Project description

The code in this repository is the result of a 3-month secondment at [Point 8 GmbH](https://point-8.de/de). This work is a part of the [SMARTHEP](https://www.smarthep.org/) ETN(European Training Network).

### Secondment summary

Point 8 is a service provider for Data Science and AI from Dortmund. They support customers comprehensively in developing digital products, from strategic consulting on data-driven business models to industry-ready implementation.

The goal of the work done during this secondment was to analyse industrial time-series data to identify and potentially extract data sections where the industrial process occured. The specific idea was to identify recurring peaking signatures in the data. The underlying process started with a down peak and finished with a up peak. Additionally, the counting of peaks from peak detection could potentially be used for fault detection of a known process.

Initially a z-score method was used to mark peaks after smoothing out the data with a simple rolling average. The approach only worked well after calibration on each dataset and could therefore not generalise very well. Instead a 1D Unet-based CNN model was used to perform segmentation on the dataset. Labeling was performed using the z-score method and some manual work.

This repository does not contain the private data used in the Point8 project, instead data is generated that has a similar signature to showcase the model.

## Code
This repository contains code for training and optimizing a 1D U-Net based model for peak detection/segmentation using *PyTorch* and *Ray Tune*. It includes data generation, model training, hyperparameter tuning, and performance evaluation.


## Requirements

- Python 3.12.7+
- PyTorch
- segmentation_models_pytorch
- scikit-learn
- numpy
- ray
- optuna

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/point8-smarthep-peak-detection.git
    cd point8-smarthep-peak-detection
    ```

2. Installing the required packages:\
    This project has been managed using [poetry](https://python-poetry.org/) and [pyenv](https://github.com/pyenv/pyenv) where the dependences can be installed using a working poetry installation:

    ```sh
    poetry install
    ```

    The dependences are documented in the `poetry.lock` and `pyproject.toml` files.

3. Jupyter Kernel

    To run the jupyter notebooks in the repository one will need a jupyter kernel. Once the poetry setup is complete a kernel can be created with:

    ```sh
    poetry run python -m ipykernel install --user --name my_kernel_name
    ```

## Usage 

### Generating the data

To generate the fake data used in this repository one needs to follow the `detection-code/fake_data_generation.ipynb` file.

### Training the model

To train the model, run the `run_training_loop.py` script:

```sh
python detection-code/run_training_loop.py
```

### Hyperparameter Optimization

To perform hyperparameter optimization using *Ray Tune*, run the `CNN_raytune_opti.py` script:

```sh 
python detection-code/CNN_raytune_opti.py 
```

## Code Overview

The training can also be done within the `CNN_training_generated_data.ipynb` file. The `detection-code/src/` folder collects common function that are used in all the scripts.

### CNN_training.py
This file contains the main training loop and data loading functions.

- `training_loop(data_loaders, config)`: Trains a U-Net model using the provided data loaders and configuration.
- `load_data(data_dir, labelType, window_size, stride, data_augmentation, n_features, num_classes, batch_size)`: Loads and preprocesses data for training, validation, and testing.
- `test_performance(test_loader, model, device)`: Evaluates the performance of the trained model on the test dataset.

### Unet_1D_model.py
This file contains the pytorch class implementation of the 1D U-Net based model.

- `build_unet(input_channels, num_classes, layer_n, conv_kernel_size, scaling_kernel_size, dropout)`: Builds the U-Net model with the specified parameters.

### data_augmentation.py
This file contains data augmentation functions.

- `augment_data(data, probability)`: Applies a series of augmentations to the input data with a given probability.

## Acknowledgements

We acknowledge funding by the European Union’s Horizon 2020 research and innovation programme, call H2020-MSCA-ITN-2020, under Grant Agreement n. 956086 (SMARTHEP).


