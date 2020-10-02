"""Tools helping with the TIMIT dataset.

Based on the version from:
https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech
"""
import soundfile as sf
import re

from os.path import join, splitext
from pathlib import Path

import numpy as np
import pandas as pd

from audio_loader.ground_truth.challenge import Challenge


PHON = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

class TimitGroundTruth(Challenge):
    """Ground truth getter for TIMIT like datasets."""

    def __init__(self, timit_like_root_folderpath, datapath="data", gtpath="data", gt_grouped_file=None):
        """Compatible with the TIMIT DARPA dataset available on kaggle.

        To use the TIMIT DARPA dataset leave the default arguments as is.
        """
        super().__init__(timit_like_root_folderpath, datapath, gtpath)

        if gt_grouped_file is None:
            df_train = pd.read_csv(join(self.root_folderpath, "train_data.csv"))
            df_train = df_train[pd.notnull(df_train['path_from_data_dir'])]
            df_test = pd.read_csv(join(self.root_folderpath, "test_data.csv"))
            df_test = df_test[pd.notnull(df_test['path_from_data_dir'])]
            self.df_all = df_train.append(df_test, ignore_index=True)
        else:
            self.df_all = pd.read_csv(join(self.root_folderpath, gt_grouped_file))
            self.df_all = self.df_all[pd.notnull(self.df_all['path_from_data_dir'])]

        # create the is_audio column if not present
        if "is_audio" not in self.df_all.keys():
            self.df_all["is_audio"] = self.df_all["path_from_data_dir"].str.match(
                ".*.wav",
                flags=re.IGNORECASE
            )

        if "is_converted_audio" in self.df_all.keys():
            self.df_all = self.df_all[np.logical_and(self.df_all["is_audio"],
                                                     self.df_all["is_converted_audio"])]
        else:
            self.df_all = self.df_all[self.df_all["is_audio"]]

        self.phon2index = {phon:index for index, phon in enumerate(PHON)}
        self.index2phn = {index:phon for index, phon in enumerate(PHON)}
        self.dict_phn_gt = get_dict_phn(join(self.root_folderpath, self.gtpath))
        self.set_gt_format()

    @property
    def training_set(self):
        """Return array of filepaths from training test."""
        return self.df_all[self.df_all["test_or_train"] == "TRAIN"]["path_from_data_dir"].values

    @property
    def testing_set(self):
        """Return array of filepaths from testing test."""
        return self.df_all[self.df_all["test_or_train"] == "TEST"]["path_from_data_dir"].values

    @property
    def gt_size(self):
        """Return the size of the ground_truth."""
        size = 0
        if self.phonetic:
            size += len(PHON)

        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            raise Exception("Speaker id not yet implemented.")

        return size

    def get_phonem_from(self, index):
        """Return the phoneme corresponding to the given index."""
        return self.index2phn[index]

    def get_index_from(self, phn):
        """Return the index corresponding to the given phoneme."""
        return self.phon2index[phn]

    def set_gt_format(self, phonetic=True, word=False, speaker_id=False):
        """Select the ground truth to show"""
        self.phonetic = phonetic
        self.word = word
        self.speaker_id = speaker_id

    def get_samples_time_in(self, filepath):
        """Return a list of tuples corresponding to the start and end times of each sample.

        Parameters:
        -----------
        filepath: str
            Filepath of the audio file we want to get the ground truth times.
        """
        audio_id = self.get_id(filepath)
        df_file = self.dict_phn_gt[audio_id]
        res_list = []
        for row in df_file.iterrows():
            res_list.append((row[1][0], row[1][1]))

        return res_list

    def get_gt_for(self, filepath):
        """Get tuples corresponding to the start, end times of each sample and
        the ground truth expected.

        Parameters:
        -----------
        filepath: str
            Filepath of the audio file we want to get the ground truth.
        """
        audio_id = self.get_id(filepath)
        df_file = self.dict_phn_gt[audio_id]
        ys = np.zeros((len(df_file.index), self.gt_size))

        res_list = []
        i = 0
        for row in df_file.iterrows():
            sample_begin, sample_end = row[1][0], row[1][1]
            self._fill_output(audio_id, sample_begin, sample_end, ys[i])
            res_list.append((sample_begin, sample_end, ys[i]))
            i += 1

        return res_list


    def _fill_output(self, id_audio, sample_begin, sample_end, output):
        """Tool to fill an output array.

        Parameters
        ----------
        id_audio: str
            id of the audio file

        sample_begin: integer > 0

        sample_end: integer > 0

        output: np.array
            Array to fill with ground truth (supposed zeros).
        """

        if self.phonetic:
            return self._fill_phon_output(id_audio, sample_begin, sample_end, output)

        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            raise Exception("Speaker id is not yet implemented.")

        raise Exception("Bad usage of set_gt_format.")

    def get_majority_gt_at_sample(self, filepath, sample_begin, sample_end):
        """Return an integer that represent the majority class for a specific sample."""
        if self.phonetic:
            return self._phon_majority(self.get_id(filepath), sample_begin, sample_end)

        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            raise Exception("Speaker id is not yet implemented.")

    def get_output_description(self):
        """Return a list that describe the output."""
        output = []
        if self.phonetic:
            output += PHON

        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            raise Exception("Speaker id is not yet implemented.")

        return output

    def _phon_majority(self, id_audio, sample_begin, sample_end):

        df_file = self.dict_phn_gt[id_audio]
        df_corresponding_time = df_file[np.logical_and(df_file["start_time"] < sample_end,
                                                       df_file["end_time"] >= sample_begin)]
        if len(df_corresponding_time) > 1:
            raise Exception("phon majority does not handle multiple labels")

        return df_corresponding_time["phn"].values

    def _fill_phon_output(self, id_audio, sample_begin, sample_end, output):
        """Tool to fill an output array.

        Parameters
        ----------
        id_audio: str
            Id of the audio file.

        sample_begin: integer > 0

        sample_end: integer > 0

        output: np.array
            Array to modify/fill with ground truth.
        """
        df_file = self.dict_phn_gt[id_audio]
        df_corresponding_time = df_file[np.logical_and(df_file["start_time"] <= sample_end,
                                                       df_file["end_time"] >= sample_begin)]
        total_samples = sample_end - sample_begin
        for row in df_corresponding_time.iterrows():
            start_frame = max(row[1][0], sample_begin)
            end_frame = min(row[1][1], sample_end)
            output[self.phon2index[row[1][2]]] += (end_frame - start_frame) / total_samples


def get_dict_phn(gt_folderpath):
    """Get dataframe of the phoneme ground truth."""
    if gt_folderpath[-1] != "/":
        gt_folderpath += "/"

    dic = {}
    for filename in Path(gt_folderpath).glob('**/*.PHN'):
        id_fn = splitext(str(filename).replace(gt_folderpath, ""))[0]
        df_file = pd.read_csv(filename, names=["start_time", "end_time", "phn"], delimiter=" ")
        dic[id_fn] = df_file

    return dic



