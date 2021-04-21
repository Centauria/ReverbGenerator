# -*- coding: utf-8 -*-
import argparse
import numpy as np
import config
import generator

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Generate reverberated signal with random room condition.'
    )
    p.add_argument('-c', '--room-count', default=12000, help='Decide how many room config will be generated.')
    p.add_argument('-p', '--items-per-room', default=4, help='Decide how many items will be generated in one room.')
    p.add_argument('--meta-output', required=True, default='dataset.csv', help='Output file name for metadata.')
    p.add_argument('--wav-output', required=True, default='dataset', help='Output waveform folder name.')
    args = p.parse_args()
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