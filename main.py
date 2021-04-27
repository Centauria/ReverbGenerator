# -*- coding: utf-8 -*-
import argparse
<<<<<<< HEAD
import numpy as np
=======
import multiprocessing
import os.path
import random
import subprocess
import time

import librosa
import pandas as pd

>>>>>>> 867efd85bc33dc33f5322d4e1028b1b45f5cb0c6
import config
import generator

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Generate reverberated signal with random room condition.'
    )
    p.add_argument('-c', '--room-count', type=int, default=12000, help='Decide how many room config will be generated.')
    p.add_argument('-p', '--items-per-room', type=int, default=4,
                   help='Decide how many items will be generated in one room.')
    p.add_argument('-t', '--tracks', type=int, default=6,
                   help='Decide how many people can be in the room simultaneously.')
    p.add_argument('-r', '--overlap-ratio', type=float, default=0.2,
                   help='Decide the expectation of the overlap ratio.\n'
                        'The ratio is of a truncated normal distribution.')
    p.add_argument('-j', '--jobs', type=int, default=1, help='Specify process count.')
    p.add_argument('-b', '--batch', type=int, default=2,
                   help='How many works should be done per circulation per worker.')
    p.add_argument('--seed', type=int, default=None, help='Specify seed.')
    p.add_argument('--timit-path', required=True, help='Specify path of TIMIT database.')
    p.add_argument('--meta-output', required=True, help='Output file name for metadata.\n'
                                                        'If exists, append to the existing file.')
    p.add_argument('--wav-output', required=True, help='Output waveform folder name.')
    args = p.parse_args()
<<<<<<< HEAD
    # TODO: main loop

    room_configs = np.zeros((args.c, 4))
    r = np.zeros((args.c, args.p))
    for i in range(args.c-1):
        room_size, source_location, mic_array_location, rt60 = config.generate_config(sample_rate=16000)
        room_configs[i, :] = [room_size, source_location, mic_array_location, rt60]
        r[i, :] = generator.make_room(room_size, source_location, mic_array_location, rt60)
    u = np.zeros()
    for i in range(args.p-1):
        u[i, :] = generator.simulate(r[i,:], input_wave, )
=======

    if args.seed:
        random.seed(args.seed)
    if os.path.isdir(args.timit_path):
        o = subprocess.check_output(['find', args.timit_path, '-name', '*.wav']).decode()
        wav_paths = o.split(os.linesep)
        wav_paths.remove('')
        wav_paths = list(map(lambda x: x.replace(args.timit_path + os.path.sep, ''), wav_paths))
    else:
        raise FileNotFoundError('TIMIT path invalid')
    if os.path.isfile(args.meta_output):
        room_configs = pd.read_csv(args.meta_output, index_col='index')
    else:
        room_configs = pd.DataFrame(columns=(
            'filename',
            'room_x', 'room_y', 'room_z',
            'src_x', 'src_y', 'src_z',
            'mic1_x', 'mic1_y', 'mic1_z',
            'mic6_x', 'mic6_y', 'mic6_z',
            'rt60', 'rt60_1', 'rt60_2', 'rt60_3',
            'rt60_4', 'rt60_5', 'rt60_6',
            'source_filename'
        ))

    os.makedirs(args.wav_output, exist_ok=True)


    def work(work_id):
        result = []
        room_size, source_location, mic_array_location, rt60 = config.generate_config(
            seed=random.randint(0, 0xFFFFFFFF),
            sample_rate=16000
        )
        print(f'Making in room {room_size}. work id {work_id}')
        room = generator.make_room(room_size, source_location, mic_array_location, rt60)
        room.compute_rir()
        actual_rt60 = room.measure_rt60()
        for j in range(args.items_per_room):
            input_wav_file = random.choice(wav_paths)
            w, sr = librosa.load(os.path.sep.join((args.timit_path, input_wav_file)), sr=None, mono=True)
            filename = '-'.join('{:.6f}'.format(time.time()).split('.')) + f'-{str(work_id).zfill(4)}.wav'
            generator.simulate(room, w,
                               to_file=os.path.sep.join((args.wav_output, filename)),
                               input_sample_rate=sr)
            result.append({
                'filename': filename,
                'room_x': room_size[0],
                'room_y': room_size[1],
                'room_z': room_size[2],
                'src_x': source_location[0],
                'src_y': source_location[1],
                'src_z': source_location[2],
                'mic1_x': mic_array_location[0, 0],
                'mic1_y': mic_array_location[1, 0],
                'mic1_z': mic_array_location[2, 0],
                'mic6_x': mic_array_location[0, -1],
                'mic6_y': mic_array_location[1, -1],
                'mic6_z': mic_array_location[2, -1],
                'rt60': rt60,
                'rt60_1': actual_rt60[0],
                'rt60_2': actual_rt60[1],
                'rt60_3': actual_rt60[2],
                'rt60_4': actual_rt60[3],
                'rt60_5': actual_rt60[4],
                'rt60_6': actual_rt60[5],
                'source_filename': input_wav_file
            })
            print(f'Generated file {filename} in room {room_size}. Using {input_wav_file}. work id {work_id}')

        return result


    def do_work(n):
        with multiprocessing.Pool(processes=args.jobs) as pool:
            results = pool.map(work, range(n))
            pool.close()
            result = pd.concat(map(lambda r: pd.DataFrame(r, index=range(len(r))), results))
            pool.join()
        return result


    rounds, remain = divmod(args.room_count, args.batch * args.jobs)
    for rn in range(rounds):
        print(f'Round {rn + 1}/{rounds}')
        room_configs = room_configs.append(do_work(args.batch * args.jobs), ignore_index=True)
    if remain > 0:
        print(f'Generating last remainders')
        room_configs = room_configs.append(do_work(remain), ignore_index=True)

    print(f'Generated {len(room_configs)} items.')
    room_configs.to_csv(args.meta_output, index_label='index', float_format='%.3f')
>>>>>>> 867efd85bc33dc33f5322d4e1028b1b45f5cb0c6
