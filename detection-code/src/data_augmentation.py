import random

import numpy as np

random.seed(10)
from scipy.interpolate import CubicSpline


def jitter(data, sigma=0.01):
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise


def scaling(data, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=data.shape)
    return data * factor


def flipping(data):
    return np.flip(data, axis=1)


def time_warp(data, sigma=0.2):
    orig_steps = np.arange(data.shape[0])
    random_warp = np.random.normal(loc=1.0, scale=sigma, size=data.shape)
    warped_steps = np.cumsum(random_warp, axis=0)
    warped_steps = warped_steps / warped_steps[-1] * (data.shape[0] - 1)
    return CubicSpline(orig_steps, data)(warped_steps)


def permutation(data, max_segments=5, seg_mode="equal"):
    """
    Permutes segments of the input data.

    Parameters:
    data (numpy.ndarray): The input data to be permuted.
    max_segments (int, optional): The maximum number of segments to divide the data into. Default is 5.
    seg_mode (str, optional): The mode of segment division. Can be "equal" for equal-sized segments or "random" for randomly sized segments. Default is "equal".

    Returns:
    numpy.ndarray: The permuted data.
    """
    orig_steps = np.arange(data.shape[0])
    num_segs = np.random.randint(1, max_segments, size=1)[0]
    if seg_mode == "random":
        split_points = np.random.choice(data.shape[0] - 2, num_segs, replace=False)
        split_points.sort()
        split_points = np.append(split_points, data.shape[0] - 1)
    else:
        split_points = np.linspace(0, data.shape[0], num=num_segs + 1, dtype=int)
    perm = np.random.permutation(num_segs)
    out = np.zeros_like(data)
    for i, seg in enumerate(perm):
        out[split_points[i] : split_points[i + 1]] = data[
            split_points[seg] : split_points[seg + 1]
        ]
    return out


def augment_data(data, probability=0.7):
    """
    Apply a series of augmentations to the input data with a given probability.
    Jitter: Adds Gaussian noise to the data.
    Scaling: Scales the data by a random factor.
    Flipping: Flips the data horizontally.
    Time_warp: Warps the data in the time dimension.
    Permutation: Permutes segments of the data.

    Parameters:
    data (numpy.ndarray): The input data to be augmented.
    probability (float, optional): The probability of applying each augmentation. Default is 0.7.

    Returns:
    numpy.ndarray: The augmented data.
    """
    augmentations = [jitter, scaling, flipping]  # TODO fix time_warp, permutation
    aug_data = data.copy()
    for aug in augmentations:
        rand = np.random.rand()
        if rand < probability:  # Apply each augmentation with probability
            aug_data = aug(aug_data)
    return aug_data
