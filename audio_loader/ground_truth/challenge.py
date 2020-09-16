"""Module for the interface of challenges."""
import abc

from math import ceil, modf
from os.path import join, splitext

import numpy as np
import soundfile as sf


class Challenge(abc.ABC):
    """Decorator for getting the ground truth and sets of a challenge."""

    def __init__(self, root_folderpath, datapath="data", gtpath="gt"):
        """Generic method.

        Parameters
        ----------
        root_folderpath: str
            Folder containing the data of a given challenge.

        datapath: str, optional
            Relative path of the data from the root_folderpath.

        gtpath: str, optional
            Relative path of ground_truth folder or file.
        """
        self.root_folderpath = root_folderpath
        self.datapath = datapath
        self.gtpath = gtpath
        if self.root_folderpath[-1] != "/":
            self.root_folderpath += "/"

        if self.datapath[-1] != "/":
            self.datapath += "/"

    @property
    def folderpath(self):
        """Return folderpath of TIMIT like dataset."""
        return self.root_folderpath

    @property
    def data_folderpath(self):
        """Return data folderpath."""
        return join(self.root_folderpath, self.datapath)

    @property
    def gt_folderpath(self):
        """Return ground truth folderpath."""
        return join(self.root_folderpath, self.gtpath)

    @property
    def training_set(self):
        """Return array of filepaths from training set."""

    @property
    def devel_set(self):
        """Return array of filepaths from development set."""

    @property
    def testing_set(self):
        """Return array of filepaths from testing set."""

    @property
    def gt_size(self):
        """Return the size of the ground_truth."""

    def get_id(self, filepath):
        """Return the id of an audio filename."""
        return splitext(splitext(filepath)[0])[0].replace(self.data_folderpath, "")


    @abc.abstractmethod
    def get_samples_time_in(self, filepath):
        """Return a list of tuples corresponding to the start and end times of each sample.

        Parameters:
        -----------
        filepath: str
            Filepath of the audio file we want to get the ground truth times.
        """

    def get_gt_from_time(self, filepath, win_size, hop_size, pad=True):
        """Get full sentence.

        Parameters
        ----------
        win_size: float
            Windows size in seconds.

        hop_size: float
            Hop size in seconds.

        pad: boolean, optional
            Use reflect padding on the original signal with length pad equal to hop_size.
        """
        sr = sf.info(join(self.data_folderpath, filepath)).samplerate
        return self.get_gt_from_sample(filepath, win_size*sr, hop_size*sr, pad)

    def get_gt_from_sample(self, filepath, win_size, hop_size, padding=True):
        """Get full sentence.

        Parameters:
        -----------
        win_size: integer > 0
            Windows size in number of samples.

        hop_size: integer > 0
            Hop size in number of samples.

        padding: boolean
            Use reflect centered  padding on the original signal like default in librosa.
        """
        frames = sf.info(join(self.data_folderpath, filepath)).frames
        id_audio = self.get_id(filepath)
        n_frames_init = 1 + int((frames - win_size) / hop_size)
        if padding:
            pad = win_size // 2
            n_frames_pad = 1 + int(frames / hop_size)  # pad*2 == win_size
            output = np.zeros((n_frames_pad, self.gt_size))
            # Fill first ground truth, while padding
            out_padding_begin = ceil(pad / hop_size)
            for i in range(out_padding_begin):
                sample_begin = hop_size * i
                reflected_samples = pad - sample_begin  # number of reflected samples taken
                self._fill_output(id_audio, 0, reflected_samples, output[i, :])
                inter = np.copy(output[i, :])  # reflect and reflected part
                rest = win_size - reflected_samples * 2
                if rest != 0:
                    # not reflected part
                    output[i, :] = np.zeros(self.gt_size)
                    self._fill_output(id_audio, reflected_samples, reflected_samples+rest, output[i, :])
                    proportion_rr = (reflected_samples / win_size) * 2 # reflect and reflected
                    output[i, :] = inter * proportion_rr + output[i, :] * (1 - proportion_rr)

            # Fill ground truth outside padding
            start_after_pad, _ = modf(pad/hop_size)
            start_after_pad = int(start_after_pad*win_size)
            end_before_pad = 1 + int((frames - win_size - start_after_pad) / hop_size) + out_padding_begin
            for i in range(out_padding_begin, end_before_pad):
                sample_begin = hop_size*(i-out_padding_begin) + start_after_pad
                self._fill_output(id_audio, sample_begin, sample_begin + win_size, output[i, :])

            # Fill lasts ground truth, while padding
            for i in range(end_before_pad, n_frames_pad):
                sample_begin = hop_size*(i-out_padding_begin) + start_after_pad
                reflected_samples = frames - pad
                inter = np.copy(output[i, :])  # no reflected
                self._fill_output(id_audio, sample_begin, reflected_samples, inter)
                proportion_nr = (reflected_samples - sample_begin) / win_size
                # reflect and reflected
                self._fill_output(id_audio, reflected_samples, frames, output[i, :])
                output[i, :] = inter * proportion_nr + output[i, :] * (1 - proportion_nr)

        else:
            output = np.zeros((n_frames_init, self.gt_size))
            for i in range(n_frames_init):
                sample_begin = hop_size*i
                self._fill_output(id_audio, sample_begin, sample_begin + win_size, output[i, :])

        return output

    @abc.abstractmethod
    def _fill_output(self, id_audio, sample_begin, sample_end, output):
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

    def get_gt_at_time(self, filepath, time_begin, time_end):
        """Get ground truth for id.

        Parameters
        ----------
        filepath: str
            Filepath of the wave file to get the ground truth.

        time_begin: float
            Time in seconds from which to start.

        time_end: float
            Time in seconds from which to end.

        Return
        ------
            One hot vector corresponding to the ground truth.
        """
        sr = sf.info(join(self.data_folderpath, filepath)).samplerate
        return self.get_gt_at_sample(filepath, int(time_begin*sr), int(time_end*sr))

    def get_gt_at_sample(self, filepath, sample_begin, sample_end):
        """Get ground truth for id.

        Parameters
        ----------
        filepath: str
            Filepath of the wave file to get the ground truth.

        sample_begin: integer
            Sample number to start with.

        sample_end: integer
            Sample number to end with.

        Return
        ------
            One hot vector corresponding to the ground truth.
        """
        # FIXME do multiple output capabilities
        output = np.zeros(self.gt_size)
        self._fill_output(self.get_id(filepath), sample_begin, sample_end, output)
        return output

    @abc.abstractmethod
    def get_output_description(self):
        """Return a list that describe the output."""
