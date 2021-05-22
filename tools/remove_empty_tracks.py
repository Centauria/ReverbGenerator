# -*- coding: utf-8 -*-
import argparse
import json
import os

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()

    for config_file in tqdm(os.listdir(args.config_path)):
        with open(os.path.join(args.config_path, config_file), 'r+') as f:
            c = json.load(f)
            k_del = []
            for k in c.keys():
                if not c[k]:  # c[k] == []
                    k_del.append(k)
            for k in k_del:
                del c[k]
            f.seek(0)
            json.dump(c, f)
            f.truncate()
