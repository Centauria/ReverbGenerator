# -*- coding: utf-8 -*-
import argparse
import json
import os

import ruamel_yaml as yaml
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()

    for config_file in tqdm(os.listdir(args.config_path)):
        a, b, c, _ = os.path.splitext(config_file)[0].split('-')
        room_file = '-'.join((a, b, c)) + '.yml'
        with open(os.path.join(args.config_path, config_file), 'r+') as f, \
                open(os.path.join(args.config_path, '..', 'rooms', room_file), 'r+') as r:
            config = json.load(f)
            k_del = []
            keys = list(config.keys())
            for k in range(len(keys)):
                if not config[keys[k]]:  # c[keys[k]] == []
                    k_del.append(k)
            for k in k_del:
                del config[keys[k]]
            f.seek(0)
            json.dump(config, f)
            f.truncate()

            room = yaml.safe_load(r)
            room_dict = dict(zip(range(len(room["sources_location"])), room["sources_location"]))
            for k in k_del:
                del room_dict[k]
            room["sources_location"] = list(room_dict.values())
            r.seek(0)
            yaml.dump(room, r)
            r.truncate()
            # Known issue: This method only fits when there's only 1 wav in a room.
