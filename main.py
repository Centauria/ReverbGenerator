# -*- coding: utf-8 -*-
import argparse
import multiprocessing
import os.path
import random

import pyroomacoustics as pra
import ruamel_yaml as yaml

import config_io
import db
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
    p.add_argument('-T', '--time', type=float, default=30.0,
                   help='Decide the total time of an item.')
    p.add_argument('-l', '--lambda', dest='poisson_lambda', type=float,
                   help='Decide the λ of the start time distribution. \n'
                        'The start times is of an exponential distribution.')
    p.add_argument('-s', '--sound-speed', type=float, default=None, help='Specify speed of sound.')
    p.add_argument('-j', '--jobs', type=int, default=1, help='Specify process count.')
    p.add_argument('-b', '--batch', type=int, default=20,
                   help='How many works should be done per circulation per worker.')
    p.add_argument('--config', type=str, default=None, help='Specify config file path.')
    p.add_argument('--seed', type=int, default=None, help='Specify seed.')
    p.add_argument('--timit-path', required=True, help='Specify path of TIMIT database.')
    p.add_argument('--split', required=True, choices=['train', 'test'], help='Choose split of dataset')
    p.add_argument('--meta-output', required=True, help='Output metadata folder name.')
    p.add_argument('--wav-output', required=True, help='Output waveform folder name.')
    p.add_argument('--prevent-repetition', action='store_true',
                   help='Add timestamp to generated filename to prevent filename repetition.')
    args = p.parse_args()

    if args.seed:
        random.seed(args.seed)
    if args.sound_speed:
        pra.constants.set('c', args.sound_speed)
    if os.path.isdir(args.timit_path):
        timit = db.TIMIT(args.timit_path)
    else:
        raise FileNotFoundError('TIMIT path invalid')

    yaml.add_representer(float, config_io.float_representer)

    room_directory = os.path.join(args.meta_output, 'rooms')
    item_directory = os.path.join(args.meta_output, 'records')
    os.makedirs(room_directory, exist_ok=True)
    os.makedirs(item_directory, exist_ok=True)
    os.makedirs(args.wav_output, exist_ok=True)


    def work(work_id):
        filename = f'{str(work_id).zfill(4)}'
        room_config = generator.generate_room_config(
            args.tracks,
            args.config,
            seed=random.randint(0, 0xFFFFFFFF)
        )
        filename_dt = config_io.save_room_config(room_config, room_directory, filename, args.prevent_repetition)
        print(f'[{work_id}] - Making in room {room_config["room_size"]}.')
        for j in range(args.items_per_room):
            track_info = generator.generate_tracks(timit, args.time, args.split, args.poisson_lambda, args.tracks)
            filename_j = filename_dt + f'-{j}'
            filename_j = config_io.save_track_info(
                track_info,
                item_directory, filename_j
            )
            generator.simulate(room_config, track_info,
                               to_file=os.path.join(args.wav_output, filename_j + '.wav'))
            print(f'[{work_id}] - Generated file {filename_j}.')
        print(f'[{work_id}] - Done.')


    def do_work(n):
        with multiprocessing.Pool(processes=args.jobs) as pool:
            pool.map(work, range(n))
            pool.close()
            pool.join()


    batches, remainder = divmod(args.room_count, args.batch)
    for i in range(batches):
        print(f'Circulation {i + 1} / {batches}')
        do_work(args.batch)
    else:
        print('Remainders ...')
        do_work(remainder)
    print(f'Generated {args.room_count} items.')
