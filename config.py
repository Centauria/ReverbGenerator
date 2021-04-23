# -*- coding: utf-8 -*-
import numpy as np


def generate_config(sample_rate=16000):
    """Generate RIR config.
    :return room_size, source_location, mic_array_location, rt60
    :rtype room_size: np.array(3,)
    :rtype source_location: np.array(3,)
    :rtype mic_array_location: np.array(3, 6)
    :rtype rt60: float
    """
    L = np.random.choice(np.linspace(3, 5, 21))
    W = np.random.choice(np.linspace(5, 10, 51))
    H = 3
    room_size = np.array([L, W, H], dtype=float)

    rt60 = np.random.choice(np.linspace(0.2, 0.8, 7))

    mic_x1 = np.random.choice(np.arange(0.5, L - 0.5 - 0.1, 0.05))
    mic_x6 = mic_x1 + 0.15
    mic_x = np.linspace(mic_x1, mic_x6, 6, dtype=float)
    mic_y = np.random.choice(np.arange(0.5, W - 0.5 + 0.05, 0.05)) * np.ones((6,), dtype=float)
    mic_z = 0.8 * np.ones((6,), dtype=float)
    mic_array_location = np.vstack((mic_x, mic_y, mic_z))

    source_location = None
    d = 0
    while d < 3.0 or d > 5.0:
        source_x = np.random.choice(np.arange(0.5, L - 0.5 + 0.05, 0.05))
        source_y = np.random.choice(np.arange(0.5, W - 0.5 + 0.05, 0.05))
        source_z = np.random.choice(np.arange(0.6, 2, 0.05))
        source_location = np.array([source_x, source_y, source_z])
        d = np.linalg.norm(source_location - mic_array_location.mean(axis=1))

    return room_size, source_location, mic_array_location, rt60
