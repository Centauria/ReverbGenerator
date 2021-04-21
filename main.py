# -*- coding: utf-8 -*-
import argparse

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
