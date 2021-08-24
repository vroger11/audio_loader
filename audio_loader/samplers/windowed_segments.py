"""Module that implement an iterator version of the sampler."""
from os.path import join
from random import shuffle

import numpy as np
import soundfile as sf

from audio_loader.samplers.decorator import SamplerBase


class WindowedSegmentSampler(SamplerBase):
    """Create samples with associated groundtruth.

    No cache involved.
    """

    def __init__(self, feature_processors, groundtruth, seg_size, overlap=0.5, supervised=True,
                 output_filepath=False, activity_detection=None):
        """Initialize the sampler.

        Parameters
        ----------
        feature_processors: list
            List of feature processors.

        groundtruth:
            Contains methods allowing to get all sets + groundtruth.

        seg_size: integer (greather than 0)
            Size of segments in number of samples.

        overlap: float between 0. and 1.
            Overlap of the segments.

        supervised: boolean
            Return the groundthruth alongside with each sample.

        activity_detection: audio_loader.actvity_detection
            Activity detection used to separate the signals, if equals to None the whole file is selected.
        """
        super().__init__(feature_processors, groundtruth,
                         supervised=supervised, output_filepath=output_filepath,
                         activity_detection=activity_detection)


        if self.fe_win_size > seg_size:
            raise Exception("seg_size should be larger or equel to feature extractors win_size")

        self.n_frames_select = 1 + int((seg_size - self.fe_win_size) / self.fe_hop_size)
        self.n_frames_hop = int(self.n_frames_select * (1 - overlap))
        if self.n_frames_hop < 1:
            raise Exception(
                f"seg_size {seg_size} is too small for the chosen extractor(s)")

        self.seg_size = self.fe_win_size + (self.n_frames_select-1) * self.fe_hop_size
        self.hop_seg_size = self.fe_win_size + (self.n_frames_hop-1) * self.fe_hop_size


    def get_samples_from(self, selected_set, randomize_files=False, file_list=None):
        """Iterator other the ground truth.

        Parameters
        ----------
        selected_set: str or list
             possible keys are: 'train', 'validation', 'test'
             if it is a list, the list represent filpaths
        """
        if isinstance(selected_set, list):
            file_list = selected_set
        else:
            file_list = self.get_file_list(selected_set)

        if randomize_files:
            file_list = file_list.copy()  # the shuffle should not impact the initial list
            shuffle(file_list)

        for filepath in file_list:
            if not isinstance(selected_set, list):
                filepath = join(self.groundtruth.data_folderpath, filepath)

            signal, sr = sf.read(filepath,
                                 always_2d=True, dtype='float32')

            x = None
            for feature_processor in self.feature_processors:
                if x is None:
                    x = feature_processor.process(signal, sr)
                else:
                    x = np.concatenate((x, feature_processor.process(signal, sr)), axis=2)

            if self.supervised:
                y = self.groundtruth.get_gt_from_sample(filepath,
                                                        self.fe_win_size,
                                                        self.fe_hop_size,
                                                        padding=self.fe_padding)

            if self.activity_detection is not None:
                fe_filter = self.activity_detection.process(signal)
                x = x[fe_filter]
                if len(x.shape) != 3:
                    x = x.reshape(1, *x.shape)

                # start FIXME y does not have a channel dimension
                fe_filter = fe_filter.all(axis=0)
                # end FIXME
                if self.supervised:
                    y = y[fe_filter]

            nb_win = x.shape[1]
            n_frames = 1 + int((nb_win - self.n_frames_select) / self.n_frames_hop)
            for i in range(0, n_frames):
                i = i*self.n_frames_hop
                x_selected = x[:, i:i+self.n_frames_select, :]
                if self.supervised:
                    y_selected = y[i:i+self.n_frames_select, :]
                    sample = (x_selected, np.sum(y_selected, axis=0)/self.n_frames_select)
                else:
                    sample = tuple(x_selected)

                if self.output_filepath:
                    yield (sample, filepath)
                else:
                    yield sample
