# -*- coding: utf-8 -*-
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruamel_yaml as yaml
import seaborn as sb
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from tqdm import tqdm


def cartesian(mic_array_location, sources_location):
    mic_array_location = np.asarray(mic_array_location)
    sources_location = np.asarray(sources_location)
    mic_center = np.average(mic_array_location, axis=1)
    xyz = sources_location - mic_center
    return xyz, mic_center


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    splits = os.listdir(os.path.join(args.path, 'wav'))
    for split in splits:
        print(f'Showing info of [{split}]:')
        wav_path = os.path.join(args.path, 'wav', split)
        record_path = os.path.join(args.path, 'meta', split, 'records')
        room_path = os.path.join(args.path, 'meta', split, 'rooms')
        print(f'wav path: {wav_path}')
        print(f'room path: {room_path}')
        print(f'record path: {record_path}')

        rooms = list(map(lambda s: os.path.splitext(s)[0], os.listdir(room_path)))
        lwh = []
        source_xyz = []
        mic_xyz = []
        rs = []
        thetas = []
        phis = []
        for room in tqdm(rooms, leave=True):
            with open(os.path.join(room_path, room + '.yml')) as f:
                room_data = yaml.safe_load(f)
            lwh.append(room_data['room_size'])
            sources = room_data['sources_location']
            source_xyz.extend(sources)
            mics = room_data['mic_array_location']
            xyz, mic_center = cartesian(mics, sources)
            mic_xyz.append(mic_center)
            r = np.linalg.norm(xyz, axis=1)
            alpha = np.arccos(xyz[:, 0] / r)
            phi = np.arctan2(xyz[:, 2], xyz[:, 1])
            rs.append(r)
            thetas.append(alpha)
            phis.append(phi)
        room_size = np.asarray(lwh)
        mic_xyz = np.asarray(mic_xyz)
        source_xyz = np.asarray(source_xyz)
        rs = np.hstack(rs)
        thetas = np.hstack(thetas)
        phis = np.hstack(phis)
        print(f'mics: {mic_xyz.shape}')
        print(f'srcs: {source_xyz.shape}')
        micf = pd.DataFrame(mic_xyz)
        micf['label'] = 'mic'
        srcf = pd.DataFrame(source_xyz)
        srcf['label'] = 'src'
        pf = pd.concat((micf, srcf), ignore_index=True)
        g = sb.FacetGrid(data=pf, hue='label', palette=dict(src='blue', mic='seagreen'), height=5)
        g.map(sb.scatterplot, 0, 1, s=1)
        g.add_legend()
        g.fig.subplots_adjust(top=0.95)
        g.ax.set_title('mic/source locations')
        plt.xlim([0, 5])
        plt.ylim([0, 10])
        plt.xlabel('dim_0')
        plt.ylabel('dim_1')
        plt.show()
        sb.histplot(rs)
        plt.title('r distribution')
        plt.xlabel(r'$r$')
        plt.show()
        hist = sb.histplot(thetas / np.pi)
        hist.xaxis.set_major_formatter(FormatStrFormatter(r'%g $\pi$'))
        hist.xaxis.set_major_locator(MultipleLocator(base=1.0))
        plt.title(r'$\theta$ distribution')
        plt.xlabel(r'$\theta$')
        plt.show()
        print(f'θ_min = {np.min(thetas)}')
        print(f'θ_max = {np.max(thetas)}')
        hist_phi = sb.histplot(phis / np.pi)
        hist_phi.xaxis.set_major_formatter(FormatStrFormatter(r'%g $\pi$'))
        hist_phi.xaxis.set_major_locator(MultipleLocator(base=0.1))
        plt.title(r'$\phi$ distribution')
        plt.xlabel(r'$\phi$')
        plt.show()
        print(f'φ_min = {np.min(phis)}')
        print(f'φ_max = {np.max(phis)}')
