# -*- coding: utf-8 -*-
import numpy as np


def generate_config(sample_rate=16000):
    """Generate RIR config.
    TODO: Fill in the function
    :return room_size, source_location, mic_array_location, rt60
    :rtype room_size: np.array(3,)
    :rtype source_location: np.array(3,)
    :rtype mic_array_location: np.array(3, 6)
    :rtype rt60: float
    """
    L = np.random.uniform(3, 5)
    W = np.random.uniform(5, 10)
    H = 3
    room_size = np.array([L, W, H], dtype=float)

    rt60 = np.random.uniform(0.2, 0.8)

    mic_x1 = np.random.uniform(0.5, L - 0.5 - 0.15)
    mic_x6 = mic_x1 + 0.15
    mic_x = np.linspace(mic_x1, mic_x6, 6, dtype=float)
    y = np.random.uniform(0.5, W - 0.5)
    mic_y = y * np.ones((6,), dtype=float)
    mic_z = 0.8 * np.ones((6,), dtype=float)
    mic_array_location = np.vstack((mic_x, mic_y, mic_z))

    source_location = None
    d = 0
    while d < 3.0 or d > 5.0:
        source_x = np.random.uniform(0.5, (L - 0.5))
        source_y = np.random.uniform(0.5, (W - 0.5))
        source_z = np.random.uniform(0.6, 1.8)
        source_location = np.array([source_x, source_y, source_z])
        d = np.linalg.norm(source_location - mic_array_location.mean(axis=1))

    return room_size, source_location, mic_array_location, rt60
