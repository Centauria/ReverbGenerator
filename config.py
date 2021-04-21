# -*- coding: utf-8 -*-
import numpy as np
import random


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

    mic_x1 = np.random.uniform(np.random.uniform(0.5, (L-0.5-0.15)))
    mic_x6 = mic_x1 + 0.15
    mic_x = np.linspace(mic_x1, mic_x6, 6, dtype=float)
    mic_y1 = np.ones((6,), dtype=int)
    y = np.random.uniform(0.5, (W-0.5))
    mic_y = np.random.uniform(0.5, (W-0.5)) * mic_y1
    mic_z = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    mic_array_location = np.vstack((mic_x, mic_y, mic_z))

    source_x = np.random.uniform(np.random.uniform(0.5, (L-0.5)))
    source_y = np.random.uniform(np.random.uniform(0.5, (W-0.5)))
    source_z = np.random.uniform(0.6, 1.8)
    d = np.sqrt(np.sum(np.abs(source_y-y)**2, np.abs(source_z-0.8)**2))
    while d not in (3, 5):
        source_y = np.random.uniform(np.random.uniform(0.5, (W - 0.5)))
    source_location = np.array([source_x, source_y, source_z], dtype=float)

    return room_size, source_location, mic_array_location, rt60
