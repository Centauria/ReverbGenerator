# -*- coding: utf-8 -*-
import os.path

import wave
import scipy.stats as stats
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
    room.sources[0].add_signal(input_wave)
    room.simulate()
    u = room.mic_array.signals
    if to_file:
        # prevent duplicated filename
        while os.path.isfile(to_file):
            to_file += '-'
        room.mic_array.to_wav(to_file, norm=True, bitdepth=np.int16)
    return u


def overlap(wave_a, wave_b, overlap_time):
    # TODO: overlap `wave_a` and `wave_b` with `overlap_time`
    """Overlap waveforms
    :returns wave_a_b
    """
    f1 = wave.open(wave_a, 'rb')
    f2 = wave.open(wave_b, 'rb')

    # 音频1的数据
    params1 = f1.getparams()
    nchannels1, sampwidth1, framerate1, nframes1, comptype1, compname1 = params1[:6]
    f1_str_data = f1.readframes(nframes1)
    f1.close()
    f1_wave_data = np.fromstring(f1_str_data, dtype=np.int16)

    # 音频2的数据
    params2 = f2.getparams()
    nchannels2, sampwidth2, framerate2, nframes2, comptype2, compname2 = params2[:6]
    f2_str_data = f2.readframes(nframes2)
    f2.close()
    f2_wave_data = np.fromstring(f2_str_data, dtype=np.int16)

    length = nframes1 + nframes2 - overlap_time

    # 零对齐补位
    temp_array1 = np.zeros((length - nframes1), dtype=np.int16)
    temp_array2 = np.zeros((length - nframes2), dtype=np.int16)
    rf1_wave_data = np.concatenate((f1_wave_data, temp_array1))
    rf2_wave_data = np.concatenate((temp_array2, f2_wave_data))

    # 合并1和2的数据
    new_wave_data = rf1_wave_data + rf2_wave_data
    wave_a_b = new_wave_data.tostring()

    return wave_a_b

def overlap_time_distribution(expectation, overlap_num):
    """Get a distribution of the overlap time
    :returns t
    """
    sigma = 1
    lower, upper = expectation - 3 * sigma, expectation + 3 * sigma
    distribution = stats.truncnorm((lower - expectation) / sigma, \
                                   (upper - expectation) / sigma, loc=expectation, scale=sigma)
    t = distribution.rvs(overlap_num)
    return t

