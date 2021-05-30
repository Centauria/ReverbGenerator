# -*- coding: utf-8 -*-
import json
import os.path
import time
from typing import Union

import numpy as np
import ruamel_yaml as yaml


def float_representer(dumper, value):
    text = '{0:.3f}'.format(value)
    return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)


def save_room_config(room_config, room_directory, filename, prevent_repitition=False):
    if prevent_repitition:
        filename = filename + '-' + '-'.join('{:.7f}'.format(time.time()).split('.'))
    with open(os.path.join(room_directory, filename + '.yml'), 'w') as f:
        yaml.dump(room_config, f)
    return filename


def save_track_info(track_info, track_directory, filename, prevent_repitition=False):
    if prevent_repitition:
        filename = filename + '-' + '-'.join('{:.7f}'.format(time.time()).split('.'))
    with open(os.path.join(track_directory, filename + '.json'), 'w') as f:
        json.dump(track_info, f)
    return filename


def conf_to_range(config: Union[dict, list, int, float]):
    if type(config) == dict:
        result = np.linspace(config['min'], config['max'], config['steps'])
    elif type(config) == list:
        result = config.copy()
    elif type(config) == float or type(config) == int:
        result = [config]
    else:
        raise ValueError('value must be dict, list, int or float')
    return result
