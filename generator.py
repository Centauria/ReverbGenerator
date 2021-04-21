# -*- coding: utf-8 -*-
import librosa
import numpy as np
import pyroomacoustics as pra


def make_room(room_size, source_location, mic_array_location, rt60, sample_rate=16000):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_size)
    r = pra.ShoeBox(
        room_size,
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order
    )
    r.add_microphone_array(mic_array_location)
    r.add_source(source_location)
    return r


def simulate(
        room: pra.ShoeBox,
        input_wave,
        to_file=None,
        input_sample_rate=16000
):
    if input_sample_rate != room.fs:
        input_wave = librosa.resample(input_wave, input_sample_rate, room.fs)
    room.sources[0].signal = input_wave
    room.simulate()
    u = room.mic_array.signals
    if to_file:
        room.mic_array.to_wav(to_file, norm=True, bitdepth=np.int16)
    return u
