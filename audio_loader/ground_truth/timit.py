"""Tools helping with the TIMIT dataset.

Based on the version from:
https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech
"""
import re

from os.path import join, splitext, dirname
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audio_loader.ground_truth.challenge import Challenge


PHON = ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',            # Stops
        'bcl', 'dcl', 'gcl', 'kcl', 'pcl', 'tcl',           # Closure
        'jh', 'ch',                                         # Affricates
        's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh',         # Fricatives
        'm', 'n', 'ng', 'em', 'en', 'eng', 'nx',            # Nasals
        'l', 'r', 'w', 'y', 'hh', 'hv', 'el',               # Semivowels and Glides
        'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay',     # Vowels
        'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er',
        'ax', 'ix', 'axr', 'ax-h',
        'pau', 'h#', 'epi' # Non-speech event
        ]

SILENCES = ['pau', 'epi', 'h#']

CLOSURES = ['bcl', 'vcl', 'dcl', 'gcl', 'kcl', 'pcl', 'tcl']

DF_PHON = pd.read_csv(join(dirname(__file__), 'timit_map.csv'), names=["original", "phon_class1", "phon_class2", "phon_class3"])


class TimitGroundTruth(Challenge):
    """Ground truth getter for TIMIT like datasets."""

    def __init__(self, timit_like_root_folderpath, datapath="data", gtpath="data", gt_grouped_file=None, with_silences=True, phon_class="original", fuse_closures=True, return_original_gt=False):
        """Compatible with the TIMIT DARPA dataset available on kaggle.

        To use the TIMIT DARPA dataset leave the default arguments as is.
        """
        super().__init__(timit_like_root_folderpath, datapath, gtpath)

        self.with_silences = with_silences
        self.phon_class = phon_class
        self.fuse_closures = fuse_closures
        self.return_original_gt = return_original_gt
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

        if self.phon_class == "original":
            self.phon2index = {phon:index for index, phon in enumerate(PHON)}
            self.index2phn = PHON
            self.silences = SILENCES
        else:
            self.index2phn = DF_PHON[self.phon_class].unique()
            # put silence at last
            self.index2phn = np.append(np.delete(self.index2phn, np.where(self.index2phn == "sil")), "sil")

            tmp_phon2index = {phon:index for index, phon in enumerate(self.index2phn)}
            # from original label to desired label
            self.phon2index = {phon:tmp_phon2index[DF_PHON.loc[DF_PHON["original"] == phon][self.phon_class].values[0]] for phon in DF_PHON["original"].unique()}
            self.silences = ["sil"]

        self.index2speaker_id = pd.unique(self.df_all["speaker_id"])
        self.speaker_id2index = {speaker_id:index for index, speaker_id in enumerate(self.index2speaker_id)}

        self.dict_gt = get_dict_gt(join(self.root_folderpath, self.gtpath), self.df_all)
        self.set_gt_format()

    @property
    def number_of_speakers(self):
        """Return the number of speakers in the Timit challenge."""
        return len(self.index2speaker_id)

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
            size += self.phon_size
        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            size += 1

        return size

    @property
    def phon_size(self):
        if self.with_silences:
            return len(self.index2phn)

        return len(self.index2phn) - len(self.silences)

    @property
    def speaker_id_size(self):
        return 1

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
        df_file, speaker_id = self.dict_gt[audio_id]
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
        df_file, speaker_id = self.dict_gt[audio_id]
        ys = np.zeros((len(df_file.index), self.gt_size))

        res_list = []
        i = 0
        if self.fuse_closures:
            previous_label = None
            previous_sample_begin = None

            for row in df_file.iterrows():
                sample_begin, sample_end = row[1][0], row[1][1]
                self._fill_output(audio_id, sample_begin, sample_end, ys[i])
                # other way to get gt label
                gt_label = row[1][2]

                if gt_label in CLOSURES:
                    previous_label = gt_label
                    previous_sample_begin = sample_begin
                else:
                    if previous_label is not None and previous_label[0] == gt_label:
                        sample_begin = previous_sample_begin

                    if self.with_silences or np.sum(ys[i]) > 0:
                        if self.return_original_gt:
                            res_list.append((sample_begin, sample_end, (ys[i], gt_label)))
                        else:
                            res_list.append((sample_begin, sample_end, ys[i]))

                    previous_label = None
                    previous_sample_begin = None

                i += 1

        else:
            for row in df_file.iterrows():
                sample_begin, sample_end = row[1][0], row[1][1]
                self._fill_output(audio_id, sample_begin, sample_end, ys[i])
                if self.with_silences or np.sum(ys[i]) > 0:
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
            self._fill_phon_output(id_audio, sample_begin, sample_end, output[:self.phon_size])

        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            output[-self.speaker_id_size] = self._get_speaker_id(id_audio)

    def get_majority_gt_at_sample(self, filepath, sample_begin, sample_end):
        """Return an integer that represent the majority class for a specific sample."""
        output = []
        if self.phonetic:
            output += self._phon_majority(self.get_id(filepath), sample_begin, sample_end)

        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            output += self._get_speaker_id(self.get_id(filepath))

        return output

    def get_output_description(self):
        """Return a list that describe the output."""
        output = []
        if self.phonetic:
            output += PHON

        if self.word:
            raise Exception("Word not yet implemented.")

        if self.speaker_id:
            output += "Speaker Id"

        return output

    def _phon_majority(self, id_audio, sample_begin, sample_end):
        df_file, speaker_id = self.dict_gt[id_audio]
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
        df_file, speaker_id = self.dict_gt[id_audio]
        df_corresponding_time = df_file[np.logical_and(df_file["start_time"] <= sample_end,
                                                       df_file["end_time"] >= sample_begin)]
        total_samples = sample_end - sample_begin
        for row in df_corresponding_time.iterrows():
            start_frame = max(row[1][0], sample_begin)
            end_frame = min(row[1][1], sample_end)
            if self.with_silences or self.phon2index[row[1][2]] < len(output):
                output[self.phon2index[row[1][2]]] += (end_frame - start_frame) / total_samples

    def _get_speaker_id(self, id_audio):
        """Tool to fill an output array.

        Parameters
        ----------
        id_audio: str
            Id of the audio file.

        output: np.array
            Array to modify/fill with ground truth.
        """
        _, speaker_id = self.dict_gt[id_audio]
        return self.speaker_id2index[speaker_id]

def get_dict_gt(gt_folderpath, df_data):
    """Get dataframe corresponding to the gt."""
    if gt_folderpath[-1] != "/":
        gt_folderpath += "/"

    dic = {}
    for filename in Path(gt_folderpath).glob('**/*.PHN'):
        id_fn = splitext(str(filename).replace(gt_folderpath, ""))[0]
        speaker_id = df_data[df_data["path_from_data_dir"].str.contains(id_fn, regex=False)]["speaker_id"].iloc[0]
        df_file = pd.read_csv(filename, names=["start_time", "end_time", "phn"], delimiter=" ")
        dic[id_fn] = df_file, speaker_id

    return dic

