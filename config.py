# -*- coding: utf-8 -*-
import numpy as np
<<<<<<< HEAD
import random
=======
>>>>>>> 867efd85bc33dc33f5322d4e1028b1b45f5cb0c6


def generate_config(seed=None, sample_rate=16000):
    """Generate RIR config.
    :return room_size, source_location, mic_array_location, rt60
    :rtype room_size: np.array(3,)
    :rtype source_location: np.array(3,)
    :rtype mic_array_location: np.array(3, 6)
    :rtype rt60: float
    """
<<<<<<< HEAD
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
=======
    if seed:
        np.random.seed(seed)

    rt60 = np.random.choice(np.linspace(0.2, 0.8, 7))

    success = False
    while not success:
        L = np.random.choice(np.linspace(3, 5, 21))
        W = np.random.choice(np.linspace(5, 10, 51))
        H = 3
        room_size = np.array([L, W, H], dtype=float)

        mic_x1 = np.random.choice(np.arange(0.5, L - 0.5 - 0.1, 0.05))
        mic_x6 = mic_x1 + 0.15
        mic_x = np.linspace(mic_x1, mic_x6, 6, dtype=float)
        mic_y = np.random.choice(np.arange(0.5, W - 0.5 + 0.05, 0.05)) * np.ones((6,), dtype=float)
        mic_z = 0.8 * np.ones((6,), dtype=float)
        mic_array_location = np.vstack((mic_x, mic_y, mic_z))

        source_location = None
        d = 0
        trial_time = 0
        while (d < 3.0 or d > 5.0) and trial_time < 10:
            source_x = np.random.choice(np.arange(0.5, L - 0.5 + 0.05, 0.05))
            source_y = np.random.choice(np.arange(0.5, W - 0.5 + 0.05, 0.05))
            source_z = np.random.choice(np.arange(0.6, 2, 0.05))
            source_location = np.array([source_x, source_y, source_z])
            d = np.linalg.norm(source_location - mic_array_location.mean(axis=1))
            trial_time += 1
        if trial_time < 10:
            success = True
>>>>>>> 867efd85bc33dc33f5322d4e1028b1b45f5cb0c6

    return room_size, source_location, mic_array_location, rt60
