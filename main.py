# -*- coding: utf-8 -*-
import argparse
import multiprocessing
import os.path
import random
import string
import subprocess

import librosa
import pandas as pd

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
    p.add_argument('-j', '--jobs', type=int, default=1, help='Specify process count.')
    p.add_argument('--timit-path', required=True, help='Specify path of TIMIT database.')
    p.add_argument('--meta-output', required=True, help='Output file name for metadata.\n'
                                                        'If exists, append to the existing file.')
    p.add_argument('--wav-output', required=True, help='Output waveform folder name.')
    args = p.parse_args()

    if os.path.isdir(args.timit_path):
        o = subprocess.check_output(['find', args.timit_path, '-name', '*.wav']).decode()
        wav_paths = o.split(os.linesep)
        wav_paths.remove('')
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
            'rt60'
        ))

    os.makedirs(args.wav_output, exist_ok=True)


    def work():
        try:
            room_size, source_location, mic_array_location, rt60 = config.generate_config(sample_rate=16000)
            room = generator.make_room(room_size, source_location, mic_array_location, rt60)
            for j in range(args.items_per_room):
                w, sr = librosa.load(random.choice(wav_paths), sr=None, mono=True)
                filename = ''.join(random.sample(string.ascii_letters + string.digits, 4)) + '.wav'
                generator.simulate(room, w,
                                   to_file=os.path.sep.join((args.wav_output, filename)),
                                   input_sample_rate=sr)
        except KeyboardInterrupt:
            pass
        finally:
            if {'filename', 'room_size', 'source_location', 'mic_array_location', 'rt60'}.issubset(dir()):
                print(f'Generated file {filename}')
                return {
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
                    'rt60': rt60
                }


    results = []
    with multiprocessing.Pool(processes=args.jobs) as pool:
        for p in range(args.room_count):
            results.append(pool.apply_async(work))
        pool.close()
        pool.join()
    room_configs = room_configs.append(
        pd.concat(list(map(lambda r: pd.DataFrame(r.get(), index=(0,)), results))),
        ignore_index=True
    )
    print(f'Generated {len(room_configs)} items.')
    room_configs.to_csv(args.meta_output, index_label='index', float_format='%.2f')
