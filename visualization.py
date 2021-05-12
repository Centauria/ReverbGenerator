# -*- coding: utf-8 -*-
import os

import pandas as pd
from bokeh.io.export import get_screenshot_as_png
from bokeh.plotting import figure


def visualize(track_info):
    df = pd.DataFrame(columns=['speaker', 'filename', 'start_time', 'end_time'])
    for speaker, wave_clips in track_info.items():
        for wc in wave_clips:
            df = df.append(dict(
                speaker=speaker,
                filename=os.path.basename(wc['wav_file']).split('.')[0],
                start_time=wc['start_time'],
                end_time=wc['end_time']
            ), ignore_index=True)
    p = figure(y_range=list(track_info.keys()), plot_height=200, plot_width=800, toolbar_location=None,
               title='Source distribution')
    p.hbar(y=df['speaker'], left=df['start_time'], right=df['end_time'], height=0.6)
    p.xaxis.axis_label = 'Time / s'
    return get_screenshot_as_png(p)
