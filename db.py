# -*- coding: utf-8 -*-
import os

import pandas as pd


class TIMIT:
    def __init__(self, timit_path: str):
        self.timit_path = timit_path
        self.data_train = pd.read_csv(
            os.path.join(timit_path, 'train_data.csv'),
            index_col='index', skip_blank_lines=True,
            usecols=[
                'index', 'speaker_id', 'filename', 'path_from_data_dir', 'is_converted_audio',
                'is_audio', 'is_word_file',
                'is_phonetic_file', 'is_sentence_file'
            ], dtype=dict(
                index='Int16',
                speaker_id=str,
                filename=str,
                path_from_data_dir=str,
                is_converted_audio='boolean',
                is_word_file='boolean',
                is_phonetic_file='boolean',
                is_sentence_file='boolean'
            ))
        self.data_test = pd.read_csv(
            os.path.join(timit_path, 'test_data.csv'),
            index_col='index', skip_blank_lines=True,
            usecols=[
                'index', 'speaker_id', 'filename', 'path_from_data_dir', 'is_converted_audio',
                'is_audio', 'is_word_file',
                'is_phonetic_file', 'is_sentence_file'
            ], dtype=dict(
                index='Int16',
                speaker_id=str,
                filename=str,
                path_from_data_dir=str,
                is_converted_audio='boolean',
                is_word_file='boolean',
                is_phonetic_file='boolean',
                is_sentence_file='boolean'
            ))
        self.data_train = self.data_train[self.data_train.index.notna()]
        self.data_test = self.data_test[self.data_test.index.notna()]

    def speakers(self, split: str):
        if split == 'train':
            data = self.data_train
        elif split == 'test':
            data = self.data_test
        else:
            raise ValueError('Split must be "train" or "test"')
        return data.speaker_id.unique()

    def audio(self, split: str, speaker_id: str):
        if split == 'train':
            data = self.data_train
        elif split == 'test':
            data = self.data_test
        else:
            raise ValueError('Split must be "train" or "test"')
        audio_list = data[data.is_converted_audio].loc[data.speaker_id == speaker_id]['path_from_data_dir'].to_list()
        audio_list = list(map(lambda s: os.path.join(self.timit_path, 'data', s), audio_list))
        return audio_list


if __name__ == '__main__':
    tmt = TIMIT('/datasets/TIMIT')
    print(tmt.audio('train', 'MMDM0'))
