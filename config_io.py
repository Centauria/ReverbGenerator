# -*- coding: utf-8 -*-
import json
import time

import ruamel_yaml as yaml


def float_representer(dumper, value):
    text = '{0:.3f}'.format(value)
    return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)


def save_room_config(room_config, filename, prevent_repitition=False):
    if prevent_repitition:
        filename = filename + '-'.join('{:.7f}'.format(time.time()).split('.'))
    with open(filename + '.yml', 'w') as f:
        yaml.dump(room_config, f)


def save_track_info(track_info, filename, prevent_repitition=False):
    if prevent_repitition:
        filename = filename + '-'.join('{:.7f}'.format(time.time()).split('.'))
    with open(filename + '.json', 'w') as f:
        json.dump(track_info, f)
