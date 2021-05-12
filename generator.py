# -*- coding: utf-8 -*-
import os.path
import wave
from collections import defaultdict, OrderedDict
from copy import deepcopy
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import ruamel_yaml as yaml

import db
from visualization import visualize


def generate_room_config(track_num: int, seed=None):
    """Generate RIR config.
    :return room_size, source_location, mic_array_location, rt60
    :rtype room_size: np.array(3,)
    :rtype source_location: np.array(3,)
    :rtype mic_array_location: np.array(3, 6)
    :rtype rt60: float
    """
    room_size, sources_location, mic_array_location = None, None, None
    if seed:
        np.random.seed(seed)

    rt60 = np.random.choice(np.linspace(0.2, 0.8, 7)).item()

    success = False
    while not success:
        L = np.random.choice(np.linspace(3, 5, 21)).item()
        W = np.random.choice(np.linspace(5, 10, 51)).item()
        H = 3
        room_size = [L, W, H]

        mic_x1 = np.random.choice(np.arange(0.5, L - 0.5 - 0.1, 0.05))
        mic_x6 = mic_x1 + 0.15
        mic_x = np.linspace(mic_x1, mic_x6, 6, dtype=float)
        mic_y = np.random.choice(np.arange(0.5, W - 0.5 + 0.05, 0.05)) * np.ones((6,), dtype=float)
        mic_z = 0.8 * np.ones((6,), dtype=float)
        mic_array_location = np.vstack((mic_x, mic_y, mic_z))

        source_location = None
        sources_location = []
        for i in range(track_num):
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
                sources_location.append(source_location)
            else:
                success = False
                break

    room_config = dict(
        room_size=room_size,
        sources_location=np.array(sources_location).tolist(),
        mic_array_location=mic_array_location.tolist(),
        rt60=rt60
    )
    return room_config


def make_room(room_size, mic_array_location, rt60, sample_rate=16000):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_size)
    r = pra.ShoeBox(
        room_size,
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order
    )
    r.add_microphone_array(mic_array_location)
    return r


def simulate(
        room_config,
        track_info,
        to_file=None
):
    assert len(room_config['sources_location']) == len(track_info)
    room = make_room(
        room_config['room_size'],
        room_config['mic_array_location'],
        room_config['rt60']
    )
    for source_location, wav_clips in zip(room_config['sources_location'], track_info.values()):
        for wav_clip in wav_clips:
            input_wave, fs = librosa.load(wav_clip['wav_file'], sr=None, mono=True)
            if fs != room.fs:
                input_wave = librosa.resample(input_wave, fs, room.fs)
            room.add_source(
                source_location,
                signal=input_wave,
                delay=wav_clip['start_time']
            )
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


def generate_tracks(dataset: db.TIMIT, total_length: float, split: str,
                    poisson_lambda: float, track_num: Optional[int] = None, speakers=None):
    track_info = defaultdict(lambda: [])
    if speakers is None:
        if track_num is not None:
            speakers = np.random.choice(dataset.speakers(split), track_num, replace=False)
        else:
            raise ValueError('track_num and speakers neither specified')
    else:
        track_num = len(speakers)
    speaker_lambda = poisson_lambda * track_num
    for p in speakers:
        wav_clips = []
        start_times = []
        while len(start_times) == 0:
            start_times = np.cumsum(np.random.exponential(speaker_lambda, max(1, int(total_length / speaker_lambda))))
        wav_files = np.random.choice(dataset.audio(split, p), len(start_times))
        for i in range(len(start_times) - 1):
            if start_times[i] <= total_length:
                end_time = start_times[i].item() + librosa.get_duration(filename=wav_files[i])
                if end_time > start_times[i + 1]:
                    start_times[i + 1] = end_time
                wav_clips.append(dict(
                    wav_file=wav_files[i].item(),
                    start_time=start_times[i].item(),
                    end_time=end_time
                ))
            else:
                break
        else:
            if start_times[-1] <= total_length:
                end_time = start_times[-1].item() + librosa.get_duration(filename=wav_files[-1])
                wav_clips.append(dict(
                    wav_file=wav_files[-1].item(),
                    start_time=start_times[-1].item(),
                    end_time=end_time
                ))
        track_info[p] = wav_clips
    return track_info


def max_end_time(track_info):
    max_end_time = 0
    for speaker, wave_clips in track_info.items():
        for wc in wave_clips:
            max_end_time = wc['end_time'] if wc['end_time'] > max_end_time else max_end_time
    return max_end_time


def event_dict(track_info):
    result = defaultdict(lambda: 0)
    for speaker, wave_clips in track_info.items():
        for wc in wave_clips:
            result[wc['start_time']] += 1
            result[wc['end_time']] -= 1
    return OrderedDict(sorted(result.items()))


def active_dict(track_info):
    edict = event_dict(track_info)
    result = OrderedDict()
    active = 0
    for k in edict.keys():
        active += edict[k]
        result[k] = active
    return result


def overlap_intervals(track_info):
    adict = active_dict(track_info)
    overlap_ranges = []
    mark = False
    for k in adict.keys():
        if not mark and adict[k] > 1:
            overlap_ranges.append(k)
            mark = True
        if mark and adict[k] <= 1:
            overlap_ranges.append(k)
            mark = False
    overlap_ranges = list(zip(overlap_ranges[::2], overlap_ranges[1::2]))
    return overlap_ranges


def empty_intervals(track_info):
    adict = active_dict(track_info)
    empty_ranges = [0]
    mark = True
    for k in adict.keys():
        if mark and adict[k] >= 1:
            empty_ranges.append(k)
            mark = False
        elif not mark and adict[k] == 0:
            empty_ranges.append(k)
            mark = True
    empty_ranges = list(zip(empty_ranges[::2], empty_ranges[1::2]))
    return empty_ranges


def ratio(track_info, intervals):
    length = max_end_time(track_info)
    time = 0
    for a, b in intervals:
        time += (b - a)
    return time / length


def shift(track_info, offset: float):
    track_info_copy = deepcopy(track_info)
    for speaker, wave_clips in track_info_copy.items():
        for wc in wave_clips:
            wc['start_time'] += offset
            wc['end_time'] += offset
    return track_info_copy


def concatenate(*track_infos):
    result = defaultdict(lambda: [])
    offset = 0
    for track_info in track_infos:
        shifted_track_info = shift(track_info, offset)
        for speaker, wave_clips in shifted_track_info.items():
            result[speaker].extend(wave_clips)
        offset += max_end_time(track_info)
    return result


def dump(track_info):
    return yaml.dump(track_info, Dumper=yaml.RoundTripDumper)


if __name__ == '__main__':
    tmt = db.TIMIT('/datasets/TIMIT')
    info = generate_tracks(tmt, 20, 'train', 0.8, track_num=6)
    info_yaml = yaml.dump(dict(info), Dumper=yaml.RoundTripDumper)
    print(info_yaml)
    img = visualize(info)
    plt.imshow(img)
